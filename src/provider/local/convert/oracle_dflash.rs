//! Numerical parity of the DFlash drafter with the transformers reference.
//!
//! Gated on `CUCA_LOCAL_DFLASH_REF`, a directory written by
//! `.omp/local-models/dflash/make_tiny_dflash.py` (not in the repository):
//! `target/` (a tiny `MuseGlimmerForConditionalGeneration` checkpoint),
//! `assistant/` (a tiny `MuseGlimmerAssistantModel` checkpoint) and
//! `ref.json` + `ref.bin` (the reference rounds). Without it the test prints
//! `SKIP:` and passes, unless `CUCA_REQUIRE_LIVE=1`.
//!
//! The reference drives transformers' own `DFlashTokenCandidateGenerator`
//! protocol by hand over several rounds with one `DFlashCache`: the target's
//! `output_hidden_states=[ids]` of the newly accepted positions are the
//! context, the pending token and the mask tokens' *raw* target embeddings
//! the block, and the assistant's last hidden rows go through the target's
//! LM head (the file also holds them under the target's logit multiplier and
//! soft-cap, which is what the engine ranks). Both checkpoints are converted
//! to F16 (their BF16 values are exact in it) and loaded in the engine's
//! exact dequantized mode, so only float rounding separates the two sides:
//! `CUCA_LOCAL_REF_TOL` (default `5e-4`) bounds `max |dlogit|` over every
//! round, and the draft tokens must be equal.

use std::path::PathBuf;

use serde_json::Value;

use super::oracle::{argmax, convert_dir, env, floats, read_gguf, skip, token_ids};
use crate::provider::local::model::cpu::dflash::DflashNet;
use crate::provider::local::model::cpu::{Hidden, Transformer};
use crate::provider::local::model::{Want, prefill_chunk};

#[test]
fn dflash_drafter_matches_the_transformers_reference() {
    let Some(dir) = env("CUCA_LOCAL_DFLASH_REF") else {
        return skip("set CUCA_LOCAL_DFLASH_REF (see make_tiny_dflash.py)");
    };
    let dir = PathBuf::from(dir);
    let tol: f32 = env("CUCA_LOCAL_REF_TOL").map_or(5e-4, |t| t.parse().unwrap());
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("ref.json")).unwrap()).unwrap();
    let hidden = usize::try_from(meta["hidden"].as_u64().unwrap()).unwrap();
    let vocab = usize::try_from(meta["vocab"].as_u64().unwrap()).unwrap();
    let block = usize::try_from(meta["block"].as_u64().unwrap()).unwrap();
    let raw = std::fs::read(dir.join("ref.bin")).unwrap();
    let reference = floats(&raw);
    let rounds = meta["rounds"].as_array().unwrap();
    // Per round: the block's final hidden rows, then the capped logits of
    // rows 1..block.
    let per_round = block * hidden + (block - 1) * vocab;
    assert_eq!(reference.len(), rounds.len() * per_round, "reference size");

    let target_path = convert_dir(dir.join("target"));
    let drafter_path = convert_dir(dir.join("assistant"));
    let longest = rounds
        .iter()
        .map(|r| r["ids"].as_array().unwrap().len())
        .max()
        .unwrap();
    let ctx = longest + block + 8;
    let mut target = Transformer::load_dequantized(&read_gguf(&target_path), ctx).unwrap();
    let mut net = DflashNet::open_dequantized(&read_gguf(&drafter_path), &target, ctx).unwrap();

    let mut worst = 0f32;
    let mut seen = 0;
    for (r, round) in rounds.iter().enumerate() {
        let seq = token_ids(&round["ids"]);
        let anchor = u32::try_from(round["anchor"].as_u64().unwrap()).unwrap();
        // The target runs the positions accepted since the last round, in
        // prefill-sized chunks, and hands their features to the drafter.
        let chunk = prefill_chunk(ctx);
        let mut pos = seen;
        while pos < seq.len() {
            let end = (pos + chunk).min(seq.len());
            let out = target
                .forward_ext(
                    &seq[pos..end],
                    pos,
                    Want::None,
                    Hidden::Layers(net.features()),
                    true,
                )
                .unwrap();
            net.observe(pos, &out.hidden.unwrap()).unwrap();
            pos = end;
        }
        seen = seq.len();

        let mut got: Vec<f32> = Vec::new();
        let drafts = net
            .draft(&target, anchor, seq.len(), block - 1, &mut |l| {
                got.extend_from_slice(l);
                Some(argmax(l))
            })
            .unwrap();
        let at = r * per_round + block * hidden;
        let want = &reference[at..at + (block - 1) * vocab];
        assert_eq!(got.len(), want.len(), "round {r}");
        let d = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "REF round {r} ({} positions): max|dlogit| = {d:.2e}",
            seq.len()
        );
        worst = worst.max(d);
        assert_eq!(
            drafts,
            token_ids(&round["draft_capped"]),
            "round {r} drafts"
        );
    }
    assert!(worst < tol, "max|dlogit| {worst:.3e} exceeds {tol:.1e}");
}
