//! The storage type of every tensor of a conversion.
//!
//! [`assign`] maps a checkpoint's [`Plan`]s to [`Dtype`]s. The uniform
//! precisions (`f16`, `q8_0`, `q4_0`) are a per-tensor table. The mixed K-quant
//! outputs (`q4_k_s`, `q4_k_m`, `q5_k_m`, `q6_k`) port llama.cpp's
//! `llama_tensor_get_type` (`src/llama-quant.cpp`, the quantizer behind
//! `llama-quantize`) for the file types and architectures the converter
//! writes: a base type per file type, promotions of the attention value
//! projections, the FFN down projections, the output head and the tied
//! embedding, the mixture-of-experts overrides, and the fallback to a type
//! with 32-value blocks when a row length is not a multiple of 256.
//!
//! What the port leaves out, because the converter has no such input:
//! the importance matrix paths, the manual type overrides and `--pure`, the
//! file types other than the four above, Falcon and the other architecture
//! checks of `llama-quant.cpp` (no converted architecture is one of them),
//! and the branch of `llama_tensor_get_type_impl` that handles a fused
//! `attn_qkv` apart from `attn_v`: llama.cpp's own `category_is_attn_v`
//! already claims `attn_qkv` and `attn_kv_b`, so that branch is unreachable
//! there and here.
//!
//! The counters follow llama.cpp's: tensors are visited in the order of its
//! tensor map (by layer, then by name; tensors outside a layer first),
//! `i_attention_wv` counts the attention value projections seen so far
//! (`n_attention_wv` is how many the file has), and `i_ffn_down` the FFN down
//! projections (`n_ffn_down` is the layer count). A mixture of experts reads the
//! layer from the tensor's name instead, because its expert stacks and shared
//! experts are several `ffn_down` tensors per layer.

use super::cfg_err;
use super::gguf_writer::Dtype;
use super::tensors::{Plan, is_matmul};
use crate::error::CucaError;
use crate::provider::local::WeightQuant;

/// What the type choice needs to know about the model besides its tensors.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ModelInfo {
    /// llama.cpp's `LLM_TYPE_70B`: a `llama` model with 80 layers and grouped
    /// key/value heads. Its attention value projections get one more bit.
    pub(crate) llama_70b: bool,
    /// The file's `<arch>.block_count` (llama.cpp's `n_layer_all`), 0 when the
    /// file has none: the layers are then counted from the tensor names.
    pub(crate) n_layer: usize,
}

/// A tensor as the type choice sees it: its name and row-major shape.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Shape<'a> {
    pub(crate) name: &'a str,
    pub(crate) dims: &'a [usize],
}

/// The storage type of each plan under `quant`, in plan order.
///
/// # Errors
///
/// As [`assign_shapes`].
pub(crate) fn assign(
    plans: &[Plan],
    quant: WeightQuant,
    info: &ModelInfo,
) -> Result<Vec<Dtype>, CucaError> {
    let shapes: Vec<Shape<'_>> = plans
        .iter()
        .map(|p| Shape {
            name: &p.name,
            dims: &p.dims,
        })
        .collect();
    assign_shapes(&shapes, quant, info)
}

/// [`assign`] over tensors given by name and shape (DeepSeek-V4.1 plans its
/// tensors without [`Plan`]s). Tensors that are not weight matrices get F32.
///
/// # Errors
///
/// [`CucaError::Config`] for a mixture of experts whose `ffn_down` tensor has
/// no `blk.<layer>.` prefix.
pub(crate) fn assign_shapes(
    shapes: &[Shape<'_>],
    quant: WeightQuant,
    info: &ModelInfo,
) -> Result<Vec<Dtype>, CucaError> {
    if !quant.is_mixed() {
        return Ok(shapes.iter().map(|s| uniform(s, quant)).collect());
    }
    Mix::new(shapes, quant, info).assign(shapes)
}

/// The storage type of a tensor at a uniform precision.
fn uniform(shape: &Shape<'_>, quant: WeightQuant) -> Dtype {
    let cols = shape.dims.last().copied().unwrap_or(1);
    if !is_matmul(shape.name, shape.dims.len()) {
        return Dtype::F32;
    }
    // llama.cpp's Q4_0 mixes keep the embedding table and the output head
    // (the tied head too) above 4 bits: the published Gemma 4 E2B Q4_0
    // has `token_embd` at Q8_0 and `per_layer_model_proj` at half
    // precision, everything else at Q4_0. `q4_0` follows it.
    let dtype = match (quant, shape.name) {
        (WeightQuant::F16, _) => Dtype::F16,
        (WeightQuant::Q4_0, "token_embd.weight" | "output.weight" | "dspine.codebook.weight") => {
            Dtype::Q8_0
        }
        (WeightQuant::Q4_0, "per_layer_model_proj.weight") => Dtype::F16,
        (WeightQuant::Q4_0, _) => Dtype::Q4_0,
        _ => Dtype::Q8_0,
    };
    if cols.is_multiple_of(dtype.block_size()) {
        dtype
    } else {
        Dtype::F32
    }
}

/// llama.cpp's `tensor_category`, the classes of tensor name the type choice
/// distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Category {
    Output,
    TokenEmbd,
    AttentionQkv,
    AttentionKvB,
    AttentionV,
    AttentionK,
    AttentionQ,
    AttentionOutput,
    FfnUp,
    FfnGate,
    FfnDown,
    Other,
}

impl Category {
    /// `tensor_get_category`.
    fn of(name: &str) -> Self {
        let has = |s: &str| name.contains(s);
        if name == "output.weight" {
            Self::Output
        } else if name == "token_embd.weight" || name == "per_layer_token_embd.weight" {
            Self::TokenEmbd
        } else if has("attn_qkv.weight") || has("attn_qkv_gate.weight") {
            Self::AttentionQkv
        } else if has("attn_kv_b.weight") {
            Self::AttentionKvB
        } else if has("attn_v.weight") {
            Self::AttentionV
        } else if has("attn_k.weight") {
            Self::AttentionK
        } else if has("attn_q.weight") {
            Self::AttentionQ
        } else if has("attn_output.weight") {
            Self::AttentionOutput
        } else if has("ffn_up") {
            Self::FfnUp
        } else if has("ffn_gate") {
            Self::FfnGate
        } else if has("ffn_down") {
            Self::FfnDown
        } else {
            Self::Other
        }
    }

    /// `category_is_attn_v`: the tensors more sensitive to quantization.
    fn is_attn_v(self) -> bool {
        matches!(
            self,
            Self::AttentionV | Self::AttentionQkv | Self::AttentionKvB
        )
    }
}

/// The layer `blk.<layer>.` of a tensor name.
fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let digits = rest.split('.').next()?;
    digits.parse().ok()
}

/// llama.cpp's `use_more_bits`: the first eighth of the layers, the last
/// eighth, and every third layer in between.
fn use_more_bits(i_layer: usize, n_layers: usize) -> bool {
    i_layer < n_layers / 8 || i_layer >= 7 * n_layers / 8 || (i_layer - n_layers / 8) % 3 == 2
}

/// The state of one pass over a model's tensors (`quantize_state_impl`).
struct Mix {
    quant: WeightQuant,
    llama_70b: bool,
    n_expert: usize,
    /// `n_layer_all`: the layers of the file, nextn blocks included.
    n_layer: usize,
    n_attention_wv: usize,
    i_attention_wv: usize,
    i_ffn_down: usize,
    /// No `output.weight`: the token embedding doubles as the head.
    tied_embeddings: bool,
}

impl Mix {
    fn new(shapes: &[Shape<'_>], quant: WeightQuant, info: &ModelInfo) -> Self {
        let n_expert = shapes
            .iter()
            .filter(|s| s.name.contains("_exps.") && s.dims.len() == 3)
            .map(|s| s.dims[0])
            .max()
            .unwrap_or(1)
            .max(1);
        Self {
            quant,
            llama_70b: info.llama_70b,
            n_expert,
            // llama.cpp's `n_layer_all` is the file's `block_count`, the
            // multi-token-prediction blocks included.
            n_layer: shapes
                .iter()
                .filter_map(|s| layer_of(s.name))
                .max()
                .map_or(0, |l| l + 1)
                .max(info.n_layer),
            n_attention_wv: shapes
                .iter()
                .filter(|s| Category::of(s.name).is_attn_v())
                .count(),
            i_attention_wv: 0,
            i_ffn_down: 0,
            tied_embeddings: !shapes.iter().any(|s| s.name == "output.weight"),
        }
    }

    /// The default type of the file type: `llama_ftype_get_default_type`.
    fn base(&self) -> Dtype {
        match self.quant {
            WeightQuant::Q5KM => Dtype::Q5K,
            WeightQuant::Q6K => Dtype::Q6K,
            _ => Dtype::Q4K,
        }
    }

    fn assign(mut self, shapes: &[Shape<'_>]) -> Result<Vec<Dtype>, CucaError> {
        // llama.cpp's tensor map orders by layer (tensors outside one first),
        // then by name; the counters depend on that order.
        let mut order: Vec<usize> = (0..shapes.len()).collect();
        order.sort_by(|&a, &b| {
            let key = |i: usize| (layer_of(shapes[i].name), shapes[i].name);
            key(a).cmp(&key(b))
        });
        let mut out = vec![Dtype::F32; shapes.len()];
        for i in order {
            out[i] = self.tensor(&shapes[i])?;
        }
        Ok(out)
    }

    /// `llama_tensor_get_type`: the type of one tensor, advancing the counters.
    fn tensor(&mut self, shape: &Shape<'_>) -> Result<Dtype, CucaError> {
        let name = shape.name;
        if !is_matmul(name, shape.dims.len()) {
            return Ok(Dtype::F32);
        }
        // `tensor_allows_quantization` excludes it: it keeps the source type
        // (half precision here, as the `q4_0` mix does).
        if name.contains("per_layer_model_proj") {
            return Ok(Dtype::F16);
        }
        let cols = shape.dims.last().copied().unwrap_or(1);
        if name == "dspine.codebook.weight" {
            return Ok(fallback(Dtype::Q8_0, cols));
        }
        let ty = self.choose(name, cols)?;
        Ok(fallback(ty, cols))
    }

    /// `llama_tensor_get_type_impl` for the four file types.
    fn choose(&mut self, name: &str, cols: usize) -> Result<Dtype, CucaError> {
        use WeightQuant::{Q4KM, Q4KS, Q5KM};
        let quant = self.quant;
        let mut ty = self.base();
        let category = Category::of(name);
        if category == Category::Output || (self.tied_embeddings && category == Category::TokenEmbd)
        {
            // The head, and the embedding that doubles as one.
            if !cols.is_multiple_of(ty.block_size()) {
                ty = Dtype::Q8_0;
            } else if ty != Dtype::Q8_0 {
                ty = Dtype::Q6K;
            }
        } else if category == Category::TokenEmbd {
            // An untied embedding keeps the base type.
        } else if category.is_attn_v() {
            if matches!(quant, Q4KM | Q5KM)
                && use_more_bits(self.i_attention_wv, self.n_attention_wv)
            {
                ty = Dtype::Q6K;
            } else if quant == Q4KS && self.i_attention_wv < 4 {
                ty = Dtype::Q5K;
            }
            if self.llama_70b && ty == Dtype::Q4K {
                // Eight query heads share each value head, so the tensor is
                // eight times smaller than `attn_q` and can afford more bits.
                ty = Dtype::Q5K;
            }
            if self.n_expert == 8 {
                ty = Dtype::Q8_0;
            }
            self.i_attention_wv += 1;
        } else if category == Category::AttentionK {
            if self.n_expert == 8 {
                ty = Dtype::Q8_0;
            }
        } else if category == Category::FfnDown {
            let i_layer = if self.n_expert > 1 {
                // A mixture of experts has several `ffn_down` tensors per
                // layer, so the layer comes from the name.
                layer_of(name)
                    .filter(|&l| l < self.n_layer)
                    .ok_or_else(|| cfg_err(format!("cannot determine the layer of `{name}`")))?
            } else {
                self.i_ffn_down
            };
            let n_layer = self.n_layer;
            match quant {
                Q4KM | Q5KM if use_more_bits(i_layer, n_layer) => ty = Dtype::Q6K,
                Q4KS if i_layer < n_layer / 8 => ty = Dtype::Q5K,
                _ => {}
            }
            self.i_ffn_down += 1;
        } else if category == Category::AttentionOutput
            && self.n_expert == 8
            && matches!(quant, Q4KS | Q4KM)
        {
            ty = Dtype::Q5K;
        }
        Ok(ty)
    }
}

/// `tensor_type_fallback`: a type with a smaller block when the row length does
/// not fill the block of `ty`, half precision when it fills no block at all.
fn fallback(ty: Dtype, cols: usize) -> Dtype {
    if cols.is_multiple_of(ty.block_size()) {
        return ty;
    }
    let smaller = match ty {
        Dtype::Q4K => Dtype::Q5_0,
        Dtype::Q5K => Dtype::Q5_1,
        Dtype::Q6K => Dtype::Q8_0,
        other => other,
    };
    if cols.is_multiple_of(smaller.block_size()) {
        smaller
    } else {
        Dtype::F16
    }
}

#[cfg(test)]
mod tests {
    use super::super::tensors::Body;
    use super::*;

    fn plan(name: &str, dims: &[usize]) -> Plan {
        Plan {
            name: name.into(),
            dims: dims.to_vec(),
            body: Body::Generated(Vec::new()),
            ops: Vec::new(),
        }
    }

    /// A llama-like model of `n` layers with every projection 256 wide:
    /// `(name, dims)` in the converter's order.
    fn dense(n: usize, tied: bool) -> Vec<Plan> {
        let mut plans = vec![
            plan("token_embd.weight", &[512, 256]),
            plan("output_norm.weight", &[256]),
        ];
        if !tied {
            plans.push(plan("output.weight", &[512, 256]));
        }
        for l in 0..n {
            for (part, dims) in [
                ("attn_norm.weight", vec![256]),
                ("attn_q.weight", vec![256, 256]),
                ("attn_k.weight", vec![64, 256]),
                ("attn_v.weight", vec![64, 256]),
                ("attn_output.weight", vec![256, 256]),
                ("ffn_norm.weight", vec![256]),
                ("ffn_gate.weight", vec![512, 256]),
                ("ffn_up.weight", vec![512, 256]),
                ("ffn_down.weight", vec![256, 512]),
            ] {
                plans.push(plan(&format!("blk.{l}.{part}"), &dims));
            }
        }
        plans
    }

    fn types(plans: &[Plan], quant: WeightQuant) -> std::collections::HashMap<String, Dtype> {
        let got = assign(plans, quant, &ModelInfo::default()).unwrap();
        plans.iter().map(|p| p.name.clone()).zip(got).collect()
    }

    fn at(map: &std::collections::HashMap<String, Dtype>, layer: usize, part: &str) -> Dtype {
        map[&format!("blk.{layer}.{part}.weight")]
    }

    #[test]
    fn use_more_bits_is_the_first_and_last_eighth_and_every_third() {
        // 32 layers: 0..4, 28..32, and 6, 9, 12, 15, 18, 21, 24, 27.
        let yes: Vec<usize> = (0..32).filter(|&i| use_more_bits(i, 32)).collect();
        assert_eq!(
            yes,
            [0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 29, 30, 31]
        );
    }

    #[test]
    fn q4_k_m_promotes_attn_v_ffn_down_and_the_head() {
        let plans = dense(32, false);
        let t = types(&plans, WeightQuant::Q4KM);
        for l in 0..32 {
            let more = use_more_bits(l, 32);
            let want = if more { Dtype::Q6K } else { Dtype::Q4K };
            assert_eq!(at(&t, l, "attn_v"), want, "attn_v {l}");
            assert_eq!(at(&t, l, "ffn_down"), want, "ffn_down {l}");
            assert_eq!(at(&t, l, "attn_q"), Dtype::Q4K);
            assert_eq!(at(&t, l, "attn_k"), Dtype::Q4K);
            assert_eq!(at(&t, l, "attn_output"), Dtype::Q4K);
            assert_eq!(at(&t, l, "ffn_gate"), Dtype::Q4K);
            assert_eq!(at(&t, l, "ffn_up"), Dtype::Q4K);
            assert_eq!(at(&t, l, "attn_norm"), Dtype::F32);
        }
        assert_eq!(t["output.weight"], Dtype::Q6K);
        // Untied: the embedding keeps the base type.
        assert_eq!(t["token_embd.weight"], Dtype::Q4K);
        assert_eq!(t["output_norm.weight"], Dtype::F32);
    }

    #[test]
    fn a_tied_embedding_takes_the_head_type() {
        let t = types(&dense(2, true), WeightQuant::Q4KM);
        assert_eq!(t["token_embd.weight"], Dtype::Q6K);
        let t = types(&dense(2, true), WeightQuant::Q5KM);
        assert_eq!(t["token_embd.weight"], Dtype::Q6K);
    }

    #[test]
    fn q5_k_m_and_q6_k_follow_the_same_table() {
        let plans = dense(32, false);
        let t = types(&plans, WeightQuant::Q5KM);
        for l in 0..32 {
            let want = if use_more_bits(l, 32) {
                Dtype::Q6K
            } else {
                Dtype::Q5K
            };
            assert_eq!(at(&t, l, "attn_v"), want);
            assert_eq!(at(&t, l, "ffn_down"), want);
            assert_eq!(at(&t, l, "ffn_up"), Dtype::Q5K);
        }
        assert_eq!(t["output.weight"], Dtype::Q6K);
        // Q6_K has no promotions: every matrix is Q6_K, the head included.
        let t = types(&plans, WeightQuant::Q6K);
        assert!(
            plans
                .iter()
                .filter(|p| is_matmul(&p.name, p.dims.len()))
                .all(|p| t[&p.name] == Dtype::Q6K)
        );
    }

    #[test]
    fn q4_k_s_promotes_the_first_values_and_the_first_eighth_of_ffn_down() {
        let t = types(&dense(32, false), WeightQuant::Q4KS);
        for l in 0..32 {
            let v = if l < 4 { Dtype::Q5K } else { Dtype::Q4K };
            assert_eq!(at(&t, l, "attn_v"), v, "attn_v {l}");
            let down = if l < 4 { Dtype::Q5K } else { Dtype::Q4K };
            assert_eq!(at(&t, l, "ffn_down"), down, "ffn_down {l}");
        }
        assert_eq!(t["output.weight"], Dtype::Q6K);
    }

    #[test]
    fn counters_follow_llama_cpps_order_not_the_plan_order() {
        // The plans reversed give the same types: layers are visited in order.
        let mut plans = dense(16, false);
        let forward = types(&plans, WeightQuant::Q4KM);
        plans.reverse();
        assert_eq!(forward, types(&plans, WeightQuant::Q4KM));
    }

    #[test]
    fn rows_that_do_not_fill_256_fall_back_as_llama_cpp_does() {
        // 96 columns: a multiple of 32, not of 256.
        let mut plans = vec![
            plan("output.weight", &[64, 96]),
            plan("blk.0.attn_q.weight", &[64, 96]),
            plan("blk.0.attn_v.weight", &[64, 96]),
            plan("blk.0.ffn_down.weight", &[96, 96]),
            // 40 columns: not even a multiple of 32.
            plan("blk.0.ffn_up.weight", &[64, 40]),
        ];
        let t = types(&plans, WeightQuant::Q4KM);
        // The head goes to Q8_0 before the table.
        assert_eq!(t["output.weight"], Dtype::Q8_0);
        // Q4_K -> Q5_0; the promoted Q6_K -> Q8_0 (one layer: all bits).
        assert_eq!(t["blk.0.attn_q.weight"], Dtype::Q5_0);
        assert_eq!(t["blk.0.attn_v.weight"], Dtype::Q8_0);
        assert_eq!(t["blk.0.ffn_down.weight"], Dtype::Q8_0);
        assert_eq!(t["blk.0.ffn_up.weight"], Dtype::F16);
        let t = types(&plans, WeightQuant::Q5KM);
        assert_eq!(t["blk.0.attn_q.weight"], Dtype::Q5_1);
        plans.truncate(2);
        let t = types(&plans, WeightQuant::Q6K);
        assert_eq!(t["blk.0.attn_q.weight"], Dtype::Q8_0);
    }

    #[test]
    fn small_tensors_stay_as_the_uniform_precisions_keep_them() {
        let plans = vec![
            plan("blk.0.ffn_gate_inp.weight", &[8, 256]),
            plan("blk.0.attn_q_norm.weight", &[256]),
            plan("blk.0.ssm_conv1d.weight", &[4, 256]),
            plan("blk.0.ssm_a", &[32]),
            plan("per_layer_model_proj.weight", &[256, 256]),
            plan("rope_freqs.weight", &[64]),
        ];
        let t = types(&plans, WeightQuant::Q4KM);
        assert_eq!(t["blk.0.ffn_gate_inp.weight"], Dtype::F32);
        assert_eq!(t["blk.0.attn_q_norm.weight"], Dtype::F32);
        assert_eq!(t["blk.0.ssm_conv1d.weight"], Dtype::F32);
        assert_eq!(t["blk.0.ssm_a"], Dtype::F32);
        assert_eq!(t["per_layer_model_proj.weight"], Dtype::F16);
        assert_eq!(t["rope_freqs.weight"], Dtype::F32);
    }

    /// A mixture of `n_expert` experts: dense attention, stacked experts and a
    /// shared expert per layer.
    fn moe(n_layer: usize, n_expert: usize) -> Vec<Plan> {
        let mut plans = vec![
            plan("token_embd.weight", &[512, 256]),
            plan("output.weight", &[512, 256]),
        ];
        for l in 0..n_layer {
            for (part, dims) in [
                ("attn_q.weight", vec![256, 256]),
                ("attn_k.weight", vec![64, 256]),
                ("attn_v.weight", vec![64, 256]),
                ("attn_output.weight", vec![256, 256]),
                ("ffn_gate_inp.weight", vec![n_expert, 256]),
                ("ffn_gate_exps.weight", vec![n_expert, 512, 256]),
                ("ffn_up_exps.weight", vec![n_expert, 512, 256]),
                ("ffn_down_exps.weight", vec![n_expert, 256, 512]),
                ("ffn_down_shexp.weight", vec![256, 512]),
            ] {
                plans.push(plan(&format!("blk.{l}.{part}"), &dims));
            }
        }
        plans
    }

    #[test]
    fn experts_read_the_layer_from_the_name() {
        // 16 experts: no Mixtral overrides; every ffn_down tensor of a layer
        // (stack and shared expert) follows the layer's own rule.
        let t = types(&moe(16, 16), WeightQuant::Q4KM);
        for l in 0..16 {
            let want = if use_more_bits(l, 16) {
                Dtype::Q6K
            } else {
                Dtype::Q4K
            };
            assert_eq!(at(&t, l, "ffn_down_exps"), want, "exps {l}");
            assert_eq!(at(&t, l, "ffn_down_shexp"), want, "shexp {l}");
            assert_eq!(at(&t, l, "ffn_gate_exps"), Dtype::Q4K);
            assert_eq!(at(&t, l, "ffn_gate_inp"), Dtype::F32);
            // attn_v counts one tensor per layer, as in the dense case.
            assert_eq!(at(&t, l, "attn_v"), want, "attn_v {l}");
        }
    }

    #[test]
    fn eight_experts_get_mixtrals_overrides() {
        let t = types(&moe(16, 8), WeightQuant::Q4KM);
        for l in 0..16 {
            assert_eq!(at(&t, l, "attn_v"), Dtype::Q8_0);
            assert_eq!(at(&t, l, "attn_k"), Dtype::Q8_0);
            assert_eq!(at(&t, l, "attn_output"), Dtype::Q5K);
            assert_eq!(at(&t, l, "attn_q"), Dtype::Q4K);
        }
        // Q5_K_M leaves the attention output alone, Q6_K too.
        let t = types(&moe(16, 8), WeightQuant::Q5KM);
        assert_eq!(at(&t, 0, "attn_output"), Dtype::Q5K);
        let t = types(&moe(16, 8), WeightQuant::Q6K);
        assert_eq!(at(&t, 0, "attn_output"), Dtype::Q6K);
        assert_eq!(at(&t, 0, "attn_v"), Dtype::Q8_0);
    }

    #[test]
    fn llama_70b_values_get_one_more_bit() {
        let plans = dense(8, false);
        let info = ModelInfo {
            llama_70b: true,
            ..Default::default()
        };
        let got = assign(&plans, WeightQuant::Q4KS, &info).unwrap();
        let t: std::collections::HashMap<_, _> =
            plans.iter().map(|p| p.name.clone()).zip(got).collect();
        // Q4_K_S: layers 0..4 are Q5_K already, the rest were Q4_K.
        for l in 0..8 {
            assert_eq!(at(&t, l, "attn_v"), Dtype::Q5K, "attn_v {l}");
        }
        assert_eq!(at(&t, 7, "attn_q"), Dtype::Q4K);
    }

    #[test]
    fn the_files_block_count_is_the_layer_total() {
        // Eight layers with tensors in a file that counts 16 blocks (trailing
        // multi-token-prediction blocks): `use_more_bits(i, 16)` picks layers
        // 0, 1, 4 and 7 among the first eight.
        let plans = dense(8, false);
        let info = ModelInfo {
            n_layer: 16,
            ..Default::default()
        };
        let got = assign(&plans, WeightQuant::Q4KM, &info).unwrap();
        let t: std::collections::HashMap<_, _> =
            plans.iter().map(|p| p.name.clone()).zip(got).collect();
        for l in 0..8 {
            let want = if [0, 1, 4, 7].contains(&l) {
                Dtype::Q6K
            } else {
                Dtype::Q4K
            };
            assert_eq!(at(&t, l, "ffn_down"), want, "ffn_down {l}");
        }
    }

    #[test]
    fn a_moe_ffn_down_without_a_layer_is_an_error() {
        let mut plans = moe(2, 16);
        plans.push(plan("ffn_down_exps.weight", &[16, 256, 512]));
        let e = assign(&plans, WeightQuant::Q4KM, &ModelInfo::default()).unwrap_err();
        assert!(e.to_string().contains("ffn_down_exps.weight"), "{e}");
    }

    #[test]
    fn uniform_precisions_are_unchanged() {
        let plans = dense(2, false);
        let q8 = assign(&plans, WeightQuant::Q8_0, &ModelInfo::default()).unwrap();
        let q4 = assign(&plans, WeightQuant::Q4_0, &ModelInfo::default()).unwrap();
        for ((p, a), b) in plans.iter().zip(q8).zip(q4) {
            if is_matmul(&p.name, p.dims.len()) {
                assert_eq!(a, Dtype::Q8_0, "{}", p.name);
                let head = p.name == "output.weight" || p.name == "token_embd.weight";
                assert_eq!(
                    b,
                    if head { Dtype::Q8_0 } else { Dtype::Q4_0 },
                    "{}",
                    p.name
                );
            } else {
                assert_eq!((a, b), (Dtype::F32, Dtype::F32), "{}", p.name);
            }
        }
    }

    #[test]
    fn dspine_codebook_is_f16_or_q8_0_across_all_precisions() {
        let plans_512 = vec![
            plan("dspine.codebook.weight", &[128, 512]),
            plan("dspine.read.weight", &[512, 256]),
            plan("dspine.gate.bias", &[64]),
        ];
        for (quant, want_codebook) in [
            (WeightQuant::F16, Dtype::F16),
            (WeightQuant::Q8_0, Dtype::Q8_0),
            (WeightQuant::Q4_0, Dtype::Q8_0),
            (WeightQuant::Q4KS, Dtype::Q8_0),
            (WeightQuant::Q4KM, Dtype::Q8_0),
            (WeightQuant::Q5KM, Dtype::Q8_0),
            (WeightQuant::Q6K, Dtype::Q8_0),
        ] {
            let got = assign(&plans_512, quant, &ModelInfo::default()).unwrap();
            assert_eq!(got[0], want_codebook, "codebook for {quant:?}");
            assert_eq!(got[2], Dtype::F32, "bias for {quant:?}");
        }

        // Unaligned code_size falls back out of Q8_0 cleanly.
        let plans_8 = vec![plan("dspine.codebook.weight", &[128, 8])];
        for (quant, want) in [
            (WeightQuant::F16, Dtype::F16),
            (WeightQuant::Q4KM, Dtype::F16),
            (WeightQuant::Q8_0, Dtype::F32),
            (WeightQuant::Q4_0, Dtype::F32),
        ] {
            let got = assign(&plans_8, quant, &ModelInfo::default()).unwrap();
            assert_eq!(got[0], want, "small codebook for {quant:?}");
        }
    }
}
