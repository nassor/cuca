//! The paged K/V cache's bookkeeping: block pools, per-sequence block tables
//! and the hash-chained prefix store. It holds no tensors; the backends
//! ([`cpu`](super::cpu), `gpu`) own the K/V data of every block, indexed by
//! the block ids this module hands out.
//!
//! # Blocks
//!
//! A *block* holds [`BLOCK_SIZE`] consecutive positions of one sequence, for
//! every K/V-owning layer of a *group*: the layers that share one attention
//! window (a [`KvGroup`]; Gemma 4's shared-K/V layers read their source's
//! block, so they join the source's group). Each group has its own
//! [`BlockPool`], so one block id means "that many positions in every layer of
//! the group". A sequence holds one *block table* ([`BlockTable`]) per group:
//! position `p` lives in `table[p / BLOCK_SIZE]`, row `p % BLOCK_SIZE`. A
//! sliding-window group's table has [`NO_BLOCK`] holes where the blocks that
//! left the window were released.
//!
//! # Prefix store
//!
//! A full block is named by a hash chain over the token ids it holds
//! (FNV-1a over the previous block's hash, then the block's 16 ids; the first
//! block chains from 0), so two sequences with the same token prefix name the
//! same blocks. [`KvPools::register`] files a sequence's full blocks under
//! their hashes; a block whose last reference is released while hashed stays
//! in the pool as *cached* (an LRU of unreferenced blocks) and is the first
//! thing an allocation evicts when no free block is left. A block without a
//! hash goes straight back to the free list. [`KvPools::lookup`] finds the
//! longest block-aligned prefix of a prompt whose blocks are still cached in
//! every group and takes a reference to each of them. The hash covers the ids
//! only (a 64-bit chain, no id comparison on a hit), which is as strong as
//! the hash: a collision would make two different prompts share K/V.
//!
//! A block is written only while its single owner is filling it; a shared or
//! cached block is read-only, because a hit covers whole blocks below the
//! prompt's last token and a sequence appends after them.

use std::collections::{HashMap, VecDeque};

use super::params::Hparams;
use crate::error::CucaError;

/// Positions per block.
pub(crate) const BLOCK_SIZE: usize = 16;

/// A table entry that holds no block: a sliding-window block that left the
/// window.
pub(crate) const NO_BLOCK: u32 = u32::MAX;

/// The blocks of one sequence in one group, by block index.
pub(crate) type BlockTable = Vec<u32>;

/// K/V-owning layers that share one attention window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KvGroup {
    /// Sliding-window length in tokens (the query included); `None` for full
    /// attention.
    pub(crate) window: Option<usize>,
    /// The K/V-owning layers of the group, ascending.
    pub(crate) layers: Vec<usize>,
}

/// One group per distinct window among the K/V-owning layers of `hp`, in
/// order of first appearance.
pub(crate) fn kv_groups(hp: &Hparams) -> Vec<KvGroup> {
    let mut groups: Vec<KvGroup> = Vec::new();
    for (il, spec) in hp.kv_owners() {
        match groups.iter_mut().find(|g| g.window == spec.window) {
            Some(g) => g.layers.push(il),
            None => groups.push(KvGroup {
                window: spec.window,
                layers: vec![il],
            }),
        }
    }
    groups
}

/// The groups of a model and the blocks each group's pool holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KvLayout {
    pub(crate) groups: Vec<KvGroup>,
    /// Pool size per group, in blocks.
    pub(crate) blocks: Vec<usize>,
    /// The group of each layer that reads K/V (owners and the layers that
    /// share an owner's cache); `None` for layers without attention.
    pub(crate) layer_group: Vec<Option<usize>>,
}

impl KvLayout {
    /// The pools of a single sequence at `context_length`: `kv_tokens` of
    /// twice the context, so a finished sequence's prefix stays cached while
    /// the next one runs.
    pub(crate) fn single(hp: &Hparams, context_length: usize) -> Self {
        Self::new(
            hp,
            context_length,
            1,
            2 * context_length,
            super::max_call_rows(context_length),
        )
    }

    /// The pools for a model of `hp` that serves at most `max_sequences`
    /// sequences over `kv_tokens` cached positions at once, `context_length`
    /// positions each, in calls of at most `max_rows` rows.
    ///
    /// A full-attention group holds `kv_tokens` rounded up to blocks. A
    /// sliding group holds what its sequences keep in the window while
    /// running, `ceil((window - 1 + max_rows) / BLOCK_SIZE) + 2` blocks each
    /// (the window, one call's rows, the block the window starts in and one
    /// more for the rows a drafter's rollback keeps), plus the blocks the
    /// cached prefixes can hold beyond a single context (`kv_tokens -
    /// context_length`), never more than the full group.
    pub(crate) fn new(
        hp: &Hparams,
        context_length: usize,
        max_sequences: usize,
        kv_tokens: usize,
        max_rows: usize,
    ) -> Self {
        let groups = kv_groups(hp);
        let full = kv_tokens.max(context_length).div_ceil(BLOCK_SIZE);
        let slack = kv_tokens
            .saturating_sub(context_length)
            .div_ceil(BLOCK_SIZE)
            .min(full);
        let blocks = groups
            .iter()
            .map(|g| match g.window {
                None => full,
                Some(w) => {
                    let per_sequence = (w.saturating_sub(1) + max_rows).div_ceil(BLOCK_SIZE) + 2;
                    (max_sequences.max(1) * per_sequence + slack).min(full)
                }
            })
            .collect();
        let mut layer_group = vec![None; hp.layers.len()];
        for (il, spec) in hp.layers.iter().enumerate() {
            layer_group[il] = groups
                .iter()
                .position(|g| g.layers.contains(&spec.kv_source));
        }
        Self {
            groups,
            blocks,
            layer_group,
        }
    }

    /// Bytes of K/V data the pools hold for `hp`: every block of a group
    /// stores [`BLOCK_SIZE`] positions of each of its layers (f32 keys and
    /// values).
    pub(crate) fn pool_bytes(&self, hp: &Hparams) -> u64 {
        self.groups
            .iter()
            .zip(&self.blocks)
            .map(|(g, &n)| {
                let per_block: u64 = g
                    .layers
                    .iter()
                    .map(|&il| {
                        let l = &hp.layers[il];
                        (2 * 4 * BLOCK_SIZE * l.n_kv_head * l.head_dim) as u64
                    })
                    .sum();
                per_block * n as u64
            })
            .sum()
    }
}

/// FNV-1a over the previous hash and the ids of one block.
pub(crate) fn chain_hash(prev: u64, ids: &[u32]) -> u64 {
    #[cfg(test)]
    if COLLIDE.get() {
        return 0x5eed;
    }
    const PRIME: u64 = 0x0100_0000_01b3;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in prev.to_le_bytes() {
        h = (h ^ u64::from(byte)).wrapping_mul(PRIME);
    }
    for id in ids {
        for byte in id.to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(PRIME);
        }
    }
    h
}

/// What names a cached block: its chain hash, the hash of the block before it
/// and its 16 token ids. A lookup hits only when all three match, so a 64-bit
/// hash collision can never make two different prefixes share K/V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockKey {
    hash: u64,
    parent: u64,
    ids: [u32; BLOCK_SIZE],
}

impl BlockKey {
    /// The key of the block holding `ids` after the block hashed `parent`
    /// (0 for the first block).
    fn new(parent: u64, ids: &[u32]) -> Self {
        let mut block = [0; BLOCK_SIZE];
        block.copy_from_slice(ids);
        Self {
            hash: chain_hash(parent, ids),
            parent,
            ids: block,
        }
    }

    /// The key of block `i` of a sequence whose chain hashes are `chain` and
    /// whose tokens are `ids`.
    fn of(chain: &[u64], ids: &[u32], i: usize) -> Self {
        let mut block = [0; BLOCK_SIZE];
        block.copy_from_slice(&ids[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE]);
        Self {
            hash: chain[i],
            parent: i.checked_sub(1).map_or(0, |p| chain[p]),
            ids: block,
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only: every chain hash is the same constant, to force collisions.
    static COLLIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
/// Counts of one pool's blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct PoolStats {
    /// Blocks no sequence references and no hash names.
    pub(crate) free: usize,
    /// Unreferenced blocks kept for their hash (evicted first-in first-out
    /// by use).
    pub(crate) cached: usize,
    /// Blocks some sequence references.
    pub(crate) used: usize,
}

/// The blocks of one group: free list, reference counts, hashes and the LRU
/// of cached blocks.
#[derive(Debug)]
pub(crate) struct BlockPool {
    /// Free block ids, popped from the back.
    free: Vec<u32>,
    refcount: Vec<u32>,
    key: Vec<Option<BlockKey>>,
    /// Unreferenced hashed blocks, least recently used at the front, with
    /// the stamp that validates the entry (a block taken out of the LRU or
    /// put back leaves its old entry stale).
    lru: VecDeque<(u32, u64)>,
    /// Stamp of each block's live LRU entry; 0 when not in the LRU.
    stamp: Vec<u64>,
    next_stamp: u64,
    cached: usize,
    by_hash: HashMap<u64, u32>,
}

impl BlockPool {
    pub(crate) fn new(blocks: usize) -> Self {
        Self {
            free: (0..blocks as u32).rev().collect(),
            refcount: vec![0; blocks],
            key: vec![None; blocks],
            lru: VecDeque::new(),
            stamp: vec![0; blocks],
            next_stamp: 1,
            cached: 0,
            by_hash: HashMap::new(),
        }
    }

    /// A block for a new owner (reference count 1): a free one, else the
    /// least recently used cached one (its hash is dropped); `None` when
    /// every block is referenced.
    pub(crate) fn alloc(&mut self) -> Option<u32> {
        if let Some(b) = self.free.pop() {
            self.refcount[b as usize] = 1;
            return Some(b);
        }
        while let Some((b, stamp)) = self.lru.pop_front() {
            let i = b as usize;
            if self.stamp[i] != stamp {
                continue;
            }
            self.stamp[i] = 0;
            self.cached -= 1;
            if let Some(k) = self.key[i].take()
                && self.by_hash.get(&k.hash) == Some(&b)
            {
                self.by_hash.remove(&k.hash);
            }
            self.refcount[i] = 1;
            return Some(b);
        }
        None
    }

    /// Drop one reference to `block`: the last one sends it to the cache when
    /// it carries a hash, else to the free list.
    pub(crate) fn release(&mut self, block: u32) {
        let i = block as usize;
        if self.refcount[i] == 0 {
            return;
        }
        self.refcount[i] -= 1;
        if self.refcount[i] > 0 {
            return;
        }
        if self.key[i].is_some() {
            self.stamp[i] = self.next_stamp;
            self.lru.push_back((block, self.next_stamp));
            self.next_stamp += 1;
            self.cached += 1;
        } else {
            self.free.push(block);
        }
    }

    /// Name `block` (referenced by its owner) by `key`, so a later
    /// [`BlockPool::lookup`] finds it. A block that is already named, or a
    /// hash another block already holds, is left alone; returns whether the
    /// block was registered.
    pub(crate) fn register(&mut self, block: u32, key: BlockKey) -> bool {
        let i = block as usize;
        if self.refcount[i] == 0 || self.key[i].is_some() || self.by_hash.contains_key(&key.hash) {
            return false;
        }
        self.key[i] = Some(key);
        self.by_hash.insert(key.hash, block);
        true
    }

    /// Whether a block named `key.hash` is held (referenced or cached) and
    /// holds the same ids after the same parent: the hash alone is 64 bits,
    /// so a collision must not make two different prefixes share K/V.
    pub(crate) fn holds(&self, key: &BlockKey) -> bool {
        self.by_hash
            .get(&key.hash)
            .is_some_and(|&b| self.key[b as usize].as_ref() == Some(key))
    }

    /// Take a reference to the block `key` names, pulling it out of the cache
    /// when nobody else held it; `None` unless [`BlockPool::holds`].
    pub(crate) fn lookup(&mut self, key: &BlockKey) -> Option<u32> {
        if !self.holds(key) {
            return None;
        }
        let &block = self.by_hash.get(&key.hash)?;
        let i = block as usize;
        if self.refcount[i] == 0 {
            self.stamp[i] = 0;
            self.cached -= 1;
        }
        self.refcount[i] += 1;
        Some(block)
    }

    /// Whether `block` carries a hash (its content is fixed).
    pub(crate) fn is_hashed(&self, block: u32) -> bool {
        self.key[block as usize].is_some()
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> PoolStats {
        PoolStats {
            free: self.free.len(),
            cached: self.cached,
            used: self.refcount.len() - self.free.len() - self.cached,
        }
    }

    /// Every block free, every hash forgotten (after a failed forward pass).
    pub(crate) fn reset(&mut self) {
        *self = Self::new(self.refcount.len());
    }
}

/// The blocks and progress of one sequence.
#[derive(Debug, Default)]
pub(crate) struct SeqBlocks {
    /// One table per group.
    pub(crate) tables: Vec<BlockTable>,
    /// Positions whose K/V the sequence holds.
    pub(crate) len: usize,
    /// Hash chain over the full blocks of the ids registered so far:
    /// `chain[i]` names block `i`.
    chain: Vec<u64>,
}

/// A prefix found in the store: `len` positions (a multiple of
/// [`BLOCK_SIZE`]) and the blocks that hold them, referenced for the caller.
#[derive(Debug)]
pub(crate) struct Hit {
    pub(crate) len: usize,
    pub(crate) tables: Vec<BlockTable>,
}

impl Hit {
    /// The hit as the start of a sequence.
    pub(crate) fn into_seq(self) -> SeqBlocks {
        let chain = Vec::with_capacity(self.len / BLOCK_SIZE);
        SeqBlocks {
            tables: self.tables,
            len: self.len,
            chain,
        }
    }
}

/// Every group's pool plus the prefix store over them.
#[derive(Debug)]
pub(crate) struct KvPools {
    groups: Vec<KvGroup>,
    pools: Vec<BlockPool>,
}

impl KvPools {
    pub(crate) fn new(layout: &KvLayout) -> Self {
        Self {
            groups: layout.groups.clone(),
            pools: layout.blocks.iter().map(|&n| BlockPool::new(n)).collect(),
        }
    }

    /// An empty sequence.
    pub(crate) fn seq(&self) -> SeqBlocks {
        SeqBlocks {
            tables: vec![Vec::new(); self.groups.len()],
            len: 0,
            chain: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> Vec<PoolStats> {
        self.pools.iter().map(BlockPool::stats).collect()
    }

    /// Give `seq` the blocks it needs for positions `0..upto`, in every
    /// group.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when a pool has no block to give; the blocks
    /// allocated so far stay in `seq`.
    pub(crate) fn reserve(&mut self, seq: &mut SeqBlocks, upto: usize) -> Result<(), CucaError> {
        let need = upto.div_ceil(BLOCK_SIZE);
        for (table, pool) in seq.tables.iter_mut().zip(&mut self.pools) {
            while table.len() < need {
                let block = pool
                    .alloc()
                    .ok_or_else(|| super::err("the K/V block pool is exhausted"))?;
                table.push(block);
            }
        }
        Ok(())
    }

    /// After `seq` accepted its first `len` positions: release the blocks of
    /// every sliding group whose positions all lie before the window of the
    /// next query (`len - (window - 1)`). With `ids` (the sequence's tokens)
    /// a block is registered in the prefix store first, so it stays cached
    /// after it leaves the window.
    pub(crate) fn release_window(&mut self, seq: &mut SeqBlocks, len: usize, ids: Option<&[u32]>) {
        if let Some(ids) = ids {
            Self::extend_chain(seq, ids, len);
        }
        for ((g, table), pool) in self
            .groups
            .iter()
            .zip(seq.tables.iter_mut())
            .zip(&mut self.pools)
        {
            let Some(w) = g.window else { continue };
            let dead = len.saturating_sub(w - 1) / BLOCK_SIZE;
            for (i, slot) in table.iter_mut().enumerate().take(dead) {
                if *slot == NO_BLOCK {
                    continue;
                }
                if let Some(ids) = ids
                    && i < seq.chain.len()
                {
                    pool.register(*slot, BlockKey::of(&seq.chain, ids, i));
                }
                pool.release(*slot);
                *slot = NO_BLOCK;
            }
        }
    }

    /// Drop the positions from `len` on and return how many `seq` retains:
    /// at most `len` and its current length. Blocks past the new end are
    /// released. The block the new end falls in must be writable and
    /// must still hold the window the new end needs: when it is named (its
    /// content is fixed) the length rounds down to its start, and when a
    /// sliding group no longer holds the blocks of the window before the new
    /// end, nothing is retained (0).
    pub(crate) fn truncate(&mut self, seq: &mut SeqBlocks, len: usize) -> usize {
        let mut len = len.min(seq.len);
        if len < seq.len {
            // A named block is read-only: keep whole blocks only.
            let tail = len / BLOCK_SIZE;
            let partial = !len.is_multiple_of(BLOCK_SIZE);
            let named = partial
                && seq.tables.iter().zip(&self.pools).any(|(t, p)| {
                    t.get(tail)
                        .is_some_and(|&b| b != NO_BLOCK && p.is_hashed(b))
                });
            if named {
                len = tail * BLOCK_SIZE;
            }
            // Every sliding group must still hold the window before `len`.
            let missing = self.groups.iter().zip(&seq.tables).any(|(g, t)| {
                g.window.is_some_and(|w| {
                    let first = len.saturating_sub(w - 1) / BLOCK_SIZE;
                    len > 0
                        && (first..len.div_ceil(BLOCK_SIZE))
                            .any(|i| t.get(i).is_none_or(|&b| b == NO_BLOCK))
                })
            });
            if missing {
                len = 0;
            }
        }
        let keep = len.div_ceil(BLOCK_SIZE);
        for (table, pool) in seq.tables.iter_mut().zip(&mut self.pools) {
            while table.len() > keep {
                if let Some(b) = table.pop().filter(|&b| b != NO_BLOCK) {
                    pool.release(b);
                }
            }
        }
        seq.len = len;
        seq.chain.truncate(len / BLOCK_SIZE);
        len
    }

    /// Register every full block of `seq` (its first `seq.len` positions
    /// hold `ids`), then release all its blocks: the full ones stay cached.
    pub(crate) fn finish(&mut self, seq: &mut SeqBlocks, ids: &[u32]) {
        let len = seq.len;
        Self::extend_chain(seq, ids, len);
        for (table, pool) in seq.tables.iter().zip(&mut self.pools) {
            for (i, &b) in table.iter().enumerate() {
                if b != NO_BLOCK && i < seq.chain.len() {
                    pool.register(b, BlockKey::of(&seq.chain, ids, i));
                }
            }
        }
        self.release(seq);
    }

    /// Release every block of `seq` without registering anything.
    pub(crate) fn release(&mut self, seq: &mut SeqBlocks) {
        for (table, pool) in seq.tables.iter_mut().zip(&mut self.pools) {
            for b in table.drain(..) {
                if b != NO_BLOCK {
                    pool.release(b);
                }
            }
        }
        seq.len = 0;
        seq.chain.clear();
    }

    /// Extend `seq.chain` over the full blocks of `ids[..len]`.
    fn extend_chain(seq: &mut SeqBlocks, ids: &[u32], len: usize) {
        let full = len.min(ids.len()) / BLOCK_SIZE;
        while seq.chain.len() < full {
            let i = seq.chain.len();
            let prev = i.checked_sub(1).map_or(0, |p| seq.chain[p]);
            seq.chain
                .push(chain_hash(prev, &ids[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE]));
        }
    }

    /// The longest block-aligned prefix of `prompt` whose K/V is cached, at
    /// most `prompt.len() - 1` positions (the last prompt token is always
    /// computed, for its logits), with a reference to each block it uses.
    ///
    /// The full-attention group must hold every block of the prefix; a
    /// sliding group only those the window before the end needs (`end -
    /// (window - 1)` on), so the prefix shortens by whole blocks until every
    /// group has them (0 when none does).
    pub(crate) fn lookup(&mut self, prompt: &[u32]) -> Hit {
        let max_blocks = prompt.len().saturating_sub(1) / BLOCK_SIZE;
        let mut keys: Vec<BlockKey> = Vec::with_capacity(max_blocks);
        let mut prev = 0;
        for chunk in prompt.as_chunks::<BLOCK_SIZE>().0.iter().take(max_blocks) {
            let key = BlockKey::new(prev, chunk);
            prev = key.hash;
            keys.push(key);
        }
        // Longest chain the full-attention group holds (no such group: any).
        let full = self.groups.iter().position(|g| g.window.is_none());
        let mut n = match full {
            Some(g) => keys.iter().take_while(|k| self.pools[g].holds(k)).count(),
            None => keys.len(),
        };
        // The first block each sliding group needs for a prefix of `n` blocks.
        let first = |w: usize, n: usize| (n * BLOCK_SIZE).saturating_sub(w - 1) / BLOCK_SIZE;
        while n > 0 {
            let ok = self.groups.iter().zip(&self.pools).all(|(g, pool)| {
                g.window
                    .is_none_or(|w| keys[first(w, n)..n].iter().all(|k| pool.holds(k)))
            });
            if ok {
                break;
            }
            n -= 1;
        }
        let mut tables = Vec::with_capacity(self.groups.len());
        for (g, pool) in self.groups.iter().zip(&mut self.pools) {
            let from = g.window.map_or(0, |w| first(w, n));
            let mut table = vec![NO_BLOCK; from];
            for key in &keys[from.min(n)..n] {
                // Present by the checks above; a miss would leave a hole the
                // attention never reads for a full group, so stop the prefix
                // rather than hand out a table with one.
                if let Some(b) = pool.lookup(key) {
                    table.push(b);
                }
            }
            tables.push(table);
        }
        Hit {
            len: n * BLOCK_SIZE,
            tables,
        }
    }

    /// Every block free, every hash forgotten (after a failed forward pass:
    /// the K/V of the running sequences is not trusted).
    pub(crate) fn reset(&mut self) {
        for pool in &mut self.pools {
            pool.reset();
        }
    }
}

/// One implicit sequence over its own pools, for the unit tests of the
/// backends: it allocates the blocks a forward call needs and, unless
/// `release` is off, releases the ones that left a sliding window (keeping the
/// rows a drafter's rollback can still return to).
#[cfg(test)]
pub(crate) struct Solo {
    pools: KvPools,
    seq: SeqBlocks,
    /// Release the blocks behind a sliding window after every call.
    pub(crate) release: bool,
}

#[cfg(test)]
impl Solo {
    pub(crate) fn new(layout: &KvLayout) -> Self {
        let pools = KvPools::new(layout);
        let seq = pools.seq();
        Self {
            pools,
            seq,
            release: true,
        }
    }

    /// Take the sequence out for a call of `n` rows at `start`, with every
    /// block the call writes.
    pub(crate) fn begin(&mut self, start: usize, n: usize) -> Result<SeqBlocks, CucaError> {
        if start != self.seq.len {
            return Err(super::err(format!(
                "forward start {start} does not match the cache length {}",
                self.seq.len
            )));
        }
        self.pools.reserve(&mut self.seq, start + n)?;
        Ok(std::mem::take(&mut self.seq))
    }

    /// Put the sequence back; `filled` is its new length when the call
    /// succeeded.
    pub(crate) fn finish(&mut self, mut seq: SeqBlocks, filled: Option<usize>) {
        if let Some(len) = filled {
            seq.len = len;
            if self.release {
                self.pools.release_window(
                    &mut seq,
                    len.saturating_sub(super::cpu::MAX_VERIFY_ROWS),
                    None,
                );
            }
        }
        self.seq = seq;
    }

    pub(crate) fn truncate(&mut self, len: usize) -> usize {
        self.pools.truncate(&mut self.seq, len)
    }

    pub(crate) fn len(&self) -> usize {
        self.seq.len
    }

    pub(crate) fn tables(&self) -> &[BlockTable] {
        &self.seq.tables
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pools(groups: &[(Option<usize>, usize)]) -> KvPools {
        let layout = KvLayout {
            groups: groups
                .iter()
                .enumerate()
                .map(|(i, &(window, _))| KvGroup {
                    window,
                    layers: vec![i],
                })
                .collect(),
            blocks: groups.iter().map(|g| g.1).collect(),
            layer_group: (0..groups.len()).map(Some).collect(),
        };
        KvPools::new(&layout)
    }

    fn ids(from: u32, n: usize) -> Vec<u32> {
        (from..from + n as u32).collect()
    }

    /// Fill a sequence with `ids` as a prefill would.
    fn fill(p: &mut KvPools, ids: &[u32]) -> SeqBlocks {
        let mut seq = p.seq();
        p.reserve(&mut seq, ids.len()).unwrap();
        seq.len = ids.len();
        seq
    }

    #[test]
    fn chain_hash_names_the_whole_prefix() {
        let a = ids(0, 16);
        let b = ids(100, 16);
        let h = chain_hash(0, &a);
        assert_eq!(h, chain_hash(0, &a));
        assert_ne!(h, chain_hash(0, &b));
        // The same ids after a different prefix are another block.
        assert_ne!(chain_hash(h, &b), chain_hash(chain_hash(0, &b), &b));
    }

    #[test]
    fn pool_allocates_free_then_evicts_the_least_recently_used() {
        let mut pool = BlockPool::new(3);
        let (a, b, c) = (
            pool.alloc().unwrap(),
            pool.alloc().unwrap(),
            pool.alloc().unwrap(),
        );
        assert!(pool.alloc().is_none());
        let ka = BlockKey::new(0, &ids(0, 16));
        let kb = BlockKey::new(0, &ids(100, 16));
        pool.register(a, ka);
        pool.register(b, kb);
        pool.release(a);
        pool.release(b);
        pool.release(c);
        // `c` carries no hash: free. `a` and `b` are cached.
        assert_eq!(
            pool.stats(),
            PoolStats {
                free: 1,
                cached: 2,
                used: 0
            }
        );
        assert_eq!(pool.alloc(), Some(c));
        // The free list is empty: the oldest cached block (`a`) goes first.
        assert_eq!(pool.alloc(), Some(a));
        assert!(!pool.holds(&ka));
        assert!(pool.holds(&kb));
    }

    /// Two different blocks whose hashes collide never share: the key also
    /// holds the parent hash and the ids, and a lookup checks all of them.
    #[test]
    fn a_hash_collision_does_not_share_blocks() {
        // At the pool: one hash, two id sets.
        let mut pool = BlockPool::new(4);
        let (a, b) = (pool.alloc().unwrap(), pool.alloc().unwrap());
        let one = BlockKey {
            hash: 9,
            parent: 0,
            ids: [1; BLOCK_SIZE],
        };
        let other_ids = BlockKey {
            ids: [2; BLOCK_SIZE],
            ..one
        };
        let other_parent = BlockKey { parent: 5, ..one };
        assert!(pool.register(a, one));
        // The hash is taken: the second block is left unregistered.
        assert!(!pool.register(b, other_ids));
        assert!(pool.holds(&one));
        assert!(!pool.holds(&other_ids));
        assert!(!pool.holds(&other_parent));
        assert_eq!(pool.lookup(&other_ids), None);
        assert_eq!(pool.lookup(&other_parent), None);
        assert_eq!(pool.lookup(&one), Some(a));

        // Through the store, with every chain hash forced equal.
        COLLIDE.set(true);
        let mut p = pools(&[(None, 8)]);
        let first = ids(0, 33);
        let mut seq = fill(&mut p, &first);
        p.finish(&mut seq, &first);
        // Same first block, different second block: with equal hashes only
        // the ids tell the second blocks apart, so only the first is shared.
        let mut prompt = first[..16].to_vec();
        prompt.extend(ids(500, 17));
        let hit = p.lookup(&prompt);
        assert_eq!(hit.len, 16);
        // A completely different prompt shares nothing.
        assert_eq!(p.lookup(&ids(900, 33)).len, 0);
        // The same prompt still hits, as far as the registered blocks go: the
        // second block lost its hash to the first (one hash, one block).
        assert_eq!(p.lookup(&first).len, 16);
        COLLIDE.set(false);
    }

    #[test]
    fn prefix_store_evicts_lru_unreferenced_blocks_only() {
        let mut p = pools(&[(None, 4)]);
        let first = ids(0, 32);
        let second = ids(1000, 32);
        let mut a = fill(&mut p, &first);
        p.finish(&mut a, &first);
        let mut b = fill(&mut p, &second);
        p.finish(&mut b, &second);
        // Four cached blocks, none free.
        assert_eq!(
            p.stats()[0],
            PoolStats {
                free: 0,
                cached: 4,
                used: 0
            }
        );
        // Re-reading the first prompt's prefix makes its blocks the most
        // recently used and references them.
        let mut prompt = first.clone();
        prompt.push(7);
        let hit = p.lookup(&prompt);
        assert_eq!(hit.len, 32);
        let mut held = hit.into_seq();
        // A new sequence needs two blocks: the second prompt's (unreferenced,
        // least recently used) are evicted, the referenced ones survive.
        let mut c = p.seq();
        p.reserve(&mut c, 32).unwrap();
        assert_eq!(p.stats()[0].used, 4);
        assert!(p.lookup(&second).len == 0, "second prompt was evicted");
        // Everything is referenced now: no block left.
        let mut d = p.seq();
        assert!(p.reserve(&mut d, 16).is_err());
        p.release(&mut c);
        p.release(&mut held);
        // The first prompt is still cached in full.
        assert_eq!(p.lookup(&prompt).len, 32);
    }

    #[test]
    fn two_prompts_sharing_a_prefix_hit_in_blocks() {
        let mut p = pools(&[(None, 16)]);
        let shared = ids(0, 40);
        let mut a_ids = shared.clone();
        a_ids.extend(ids(500, 10));
        let mut a = fill(&mut p, &a_ids);
        p.finish(&mut a, &a_ids);
        // 50 positions: three full blocks cached, the trailing 2 are not.
        let mut b_ids = shared.clone();
        b_ids.extend(ids(900, 20));
        let hit = p.lookup(&b_ids);
        // Shares 40 tokens: the two full blocks (32) of the shared prefix.
        assert_eq!(hit.len, 32);
        assert_eq!(hit.tables[0].len(), 2);
        // The same prompt again: all but its last token, in whole blocks.
        let all = p.lookup(&a_ids);
        assert_eq!(all.len, 48);
        // The prompt that is exactly 3 blocks reads 2 (its last token is
        // always computed).
        let exact = p.lookup(&a_ids[..48]);
        assert_eq!(exact.len, 32);
        assert_eq!(p.lookup(&a_ids[..16]).len, 0);
    }

    #[test]
    fn sliding_group_releases_old_blocks() {
        // One full and one sliding group (window 20: the next query needs
        // the 19 positions before it).
        let mut p = pools(&[(None, 16), (Some(20), 16)]);
        let toks = ids(0, 96);
        let mut seq = fill(&mut p, &toks);
        assert_eq!(seq.tables[1].len(), 6);
        p.release_window(&mut seq, 96, Some(&toks));
        // Positions before 96 - 19 = 77 are dead: blocks 0..4 (positions
        // below 64); block 4 (64..80) still holds 77..80.
        assert_eq!(&seq.tables[1][..4], &[NO_BLOCK; 4]);
        assert!(seq.tables[1][4] != NO_BLOCK && seq.tables[1][5] != NO_BLOCK);
        assert_eq!(seq.tables[0].iter().filter(|&&b| b == NO_BLOCK).count(), 0);
        // The released blocks went to the cache (they were registered).
        assert_eq!(p.stats()[1].cached, 4);
        // Truncating to a length whose window was released retains nothing;
        // one inside the live window is kept (the window of 90 starts at 71,
        // in block 4; the window of 80 starts at 61, in the released block 3).
        assert_eq!(p.truncate(&mut seq, 40), 0);
        let mut seq2 = fill(&mut p, &toks);
        p.release_window(&mut seq2, 96, None);
        assert_eq!(p.truncate(&mut seq2, 90), 90);
        assert_eq!(seq2.tables[1].len(), 6);
        assert_eq!(p.truncate(&mut seq2, 80), 0);
        assert!(seq2.tables.iter().all(Vec::is_empty));
    }

    #[test]
    fn a_sliding_group_limits_the_prefix_hit_to_its_window() {
        let mut p = pools(&[(None, 16), (Some(20), 6)]);
        let toks = ids(0, 96);
        let mut seq = fill(&mut p, &toks);
        p.release_window(&mut seq, 96, Some(&toks));
        p.finish(&mut seq, &toks);
        let mut prompt = toks.clone();
        prompt.push(5);
        let hit = p.lookup(&prompt);
        // All six blocks are cached in both groups (registered while the
        // window moved): the whole 96 is a hit; the sliding table holds
        // only the blocks its window needs.
        assert_eq!(hit.len, 96);
        assert_eq!(hit.tables[0].len(), 6);
        assert_eq!(hit.tables[1].len(), 6);
        assert_eq!(&hit.tables[1][..4], &[NO_BLOCK; 4]);
        // Evict the sliding group's old blocks (its pool has 6 blocks: five
        // new blocks displace the cached ones from the front).
        let mut h = hit.into_seq();
        p.release(&mut h);
        let mut other = p.seq();
        p.reserve(&mut other, 16 * 5).unwrap();
        // Some sliding blocks are gone; the hit shortens to what the window
        // of a shorter prefix finds, never past it.
        let again = p.lookup(&prompt);
        assert!(again.len < 96);
        assert_eq!(again.len % BLOCK_SIZE, 0);
    }

    #[test]
    fn truncate_keeps_named_blocks_whole() {
        let mut p = pools(&[(None, 8)]);
        let toks = ids(0, 48);
        let mut seq = fill(&mut p, &toks);
        p.finish(&mut seq, &toks);
        let mut prompt = toks.clone();
        prompt.push(1);
        let mut held = p.lookup(&prompt).into_seq();
        assert_eq!(held.len, 48);
        // Inside a named block: rounds down to the block start.
        assert_eq!(p.truncate(&mut held, 40), 32);
        assert_eq!(held.tables[0].len(), 2);
        // On a boundary the blocks are untouched.
        assert_eq!(p.truncate(&mut held, 32), 32);
    }
}
