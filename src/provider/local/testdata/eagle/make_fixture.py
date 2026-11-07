"""Regenerates the tiny EAGLE-drafter oracle fixture next to this file.

    python make_fixture.py

writes `params.json` and `consolidated.safetensors` (a tiny drafter in
Mistral's own checkpoint format: FP8 e4m3 weights, a BF16 `qscale_weight` and
`qscale_act` per matrix, BF16 norms) and `reference.json` (what a transformers
`Mistral4Model` computes from those weights). The Rust side
(`model/cpu/eagle.rs`, test `the_converted_drafter_matches_the_transformers_reference`)
converts the checkpoint with the engine's own converter, loads it next to a
tiny `mistral4` model whose embedding and LM head are `embd` and `lm_head`
below, and compares the drafter's draft logits with this file's.

The reference follows vLLM's `EagleMistralLarge3Model` (the drafter's runtime
for `mistralai/Mistral-Small-4-119B-2603-eagle`): the input of position `j` is
`fc(cat(embed(t[j + 1]), h[j]))` (embedding first, no norm), then two dense
Mistral4 decoder layers at RoPE position `j`, then the drafter's final norm;
the logits come from the target's LM head; the normed output is the next draft
step's hidden row. Requires torch, transformers (with `mistral4`) and
safetensors.

The `controls` are the same chain with one convention changed (hidden row
first, no Llama 4 query scaling, RoPE position `j + 1`): the Rust test checks
that the engine's logits are far from every one of them, so the comparison
would notice each of those mistakes.
"""

import json
import pathlib

import torch
from safetensors.torch import save_file
from transformers import Mistral4Config, Mistral4Model

HERE = pathlib.Path(__file__).parent

# The tiny geometry of the engine's `tiny::write_mistral4` target.
DIM, LAYERS, HEADS = 64, 2, 4
FF, VOCAB = 96, 16
Q_LORA, KV_LORA, NOPE, ROT, V_HEAD = 32, 16, 8, 8, 16
FACTOR, ORIG, BETA_FAST, BETA_SLOW = 2, 8, 32, 1
LLAMA4_BETA = 0.3
POSITIONS = 10  # target positions whose hidden rows the drafter sees
CHAIN = 4  # draft steps


def params():
    return {
        "dim": DIM, "n_layers": LAYERS, "head_dim": V_HEAD, "hidden_dim": FF,
        "n_heads": HEADS, "n_kv_heads": HEADS, "rope_theta": 10000.0,
        "norm_eps": 1e-06, "vocab_size": VOCAB, "tied_embeddings": False,
        "max_position_embeddings": ORIG * FACTOR,
        "llama_4_scaling": {"original_max_position_embeddings": ORIG, "beta": LLAMA4_BETA},
        "q_lora_rank": Q_LORA, "qk_rope_head_dim": ROT, "qk_nope_head_dim": NOPE,
        "kv_lora_rank": KV_LORA, "v_head_dim": V_HEAD,
        "quantization": {"qformat_weight": "fp8_e4m3", "qscheme_act": "TENSOR"},
        "yarn": {"original_max_position_embeddings": ORIG, "factor": FACTOR,
                 "apply_scale": False, "beta": BETA_FAST, "alpha": BETA_SLOW},
        "moe": None, "vision_encoder": None,
    }


def hf_config(llama4_beta=LLAMA4_BETA):
    return Mistral4Config(
        vocab_size=VOCAB, hidden_size=DIM, intermediate_size=FF, num_hidden_layers=LAYERS,
        num_attention_heads=HEADS, num_key_value_heads=HEADS, kv_lora_rank=KV_LORA,
        q_lora_rank=Q_LORA, qk_nope_head_dim=NOPE, qk_rope_head_dim=ROT, v_head_dim=V_HEAD,
        head_dim=NOPE + ROT, qk_head_dim=NOPE + ROT, first_k_dense_replace=LAYERS,
        n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=FF,
        n_shared_experts=1, n_group=1, topk_group=1, norm_topk_prob=True,
        routed_scaling_factor=1.0, rms_norm_eps=1e-6,
        max_position_embeddings=ORIG * FACTOR,
        rope_parameters={
            "type": "yarn", "rope_type": "yarn", "rope_theta": 10000.0,
            "factor": float(FACTOR), "original_max_position_embeddings": ORIG,
            "beta_fast": float(BETA_FAST), "beta_slow": float(BETA_SLOW),
            "mscale_all_dim": 1.0, "mscale": 1.0, "llama_4_scaling_beta": llama4_beta,
            "partial_rotary_factor": ROT / (NOPE + ROT),
        },
        tie_word_embeddings=False, attn_implementation="eager",
    )


def make_weights(gen):
    """Random weights; every norm is randomised so a missing or misplaced norm shows."""
    def mat(rows, cols, scale=0.3):
        return torch.randn(rows, cols, generator=gen) * scale

    def norm(n):
        return 1.0 + 0.25 * torch.randn(n, generator=gen)

    w = {"eagle_linear": mat(DIM, 2 * DIM, 0.2), "norm": norm(DIM)}
    for il in range(LAYERS):
        p = f"layers.{il}."
        w[p + "attention_norm"] = norm(DIM)
        w[p + "ffn_norm"] = norm(DIM)
        w[p + "attention.q_a_norm"] = norm(Q_LORA)
        w[p + "attention.kv_a_norm"] = norm(KV_LORA)
        w[p + "attention.wq_a"] = mat(Q_LORA, DIM)
        w[p + "attention.wq_b"] = mat(HEADS * (NOPE + ROT), Q_LORA)
        w[p + "attention.wkv_a_with_mqa"] = mat(KV_LORA + ROT, DIM)
        w[p + "attention.wkv_b"] = mat(HEADS * (NOPE + V_HEAD), KV_LORA, 0.5)
        w[p + "attention.wo"] = mat(DIM, HEADS * V_HEAD)
        w[p + "feed_forward.w1"] = mat(FF, DIM)
        w[p + "feed_forward.w2"] = mat(DIM, FF)
        w[p + "feed_forward.w3"] = mat(FF, DIM)
    return w


def package(weights, gen):
    """The checkpoint file: matrices as FP8 e4m3 with BF16 scalar scales, norms as BF16.

    Returns the safetensors tensors and the f32 values the engine will see: the
    FP8 value times its scale, rounded to f16 (the converter's `f16` output) for
    a matrix, the BF16 value for a norm.
    """
    file, seen = {}, {}
    for name, v in weights.items():
        if v.ndim == 1:
            file[name + ".weight"] = v.to(torch.bfloat16)
            seen[name] = file[name + ".weight"].float()
            continue
        scale = (v.abs().max() / 448.0).to(torch.bfloat16)
        file[name + ".weight"] = (v / scale.float()).to(torch.float8_e4m3fn)
        file[name + ".qscale_weight"] = scale
        file[name + ".qscale_act"] = (torch.rand((), generator=gen) + 0.5).to(torch.bfloat16)
        seen[name] = (file[name + ".weight"].float() * scale.float()).to(torch.float16).float()
    return file, seen


def hf_state(seen):
    sd = {"norm.weight": seen["norm"]}
    names = {
        "attention_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
        "attention.q_a_norm": "self_attn.q_a_layernorm",
        "attention.kv_a_norm": "self_attn.kv_a_layernorm",
        "attention.wq_a": "self_attn.q_a_proj", "attention.wq_b": "self_attn.q_b_proj",
        "attention.wkv_a_with_mqa": "self_attn.kv_a_proj_with_mqa",
        "attention.wkv_b": "self_attn.kv_b_proj", "attention.wo": "self_attn.o_proj",
        "feed_forward.w1": "mlp.gate_proj", "feed_forward.w2": "mlp.down_proj",
        "feed_forward.w3": "mlp.up_proj",
    }
    for il in range(LAYERS):
        for mistral, hf in names.items():
            sd[f"layers.{il}.{hf}.weight"] = seen[f"layers.{il}.{mistral}"]
    return sd


def build_model(seen, llama4_beta=LLAMA4_BETA):
    model = Mistral4Model(hf_config(llama4_beta)).float().eval()
    missing, unexpected = model.load_state_dict(hf_state(seen), strict=False)
    assert not unexpected, unexpected
    assert set(missing) == {"embed_tokens.weight"}, missing
    return model


@torch.no_grad()
def drafter(model, fc, embeds, hiddens, shift=0, hidden_first=False):
    """Normed drafter output at every pair: pair `i` sits at RoPE position `i + shift`."""
    parts = [hiddens, embeds] if hidden_first else [embeds, hiddens]
    x = torch.cat(parts, dim=-1) @ fc.T
    positions = torch.arange(x.shape[0]) + shift
    out = model(inputs_embeds=x[None], position_ids=positions[None]).last_hidden_state
    return out[0]


def main():
    for seed in range(1, 200):
        gen = torch.Generator().manual_seed(seed)
        weights = make_weights(gen)
        file, seen = package(weights, gen)
        embd = torch.randn(VOCAB, DIM, generator=gen) * 0.5
        lm_head = torch.randn(VOCAB, DIM, generator=gen) * 0.5
        tokens = torch.randint(3, VOCAB, (POSITIONS,), generator=gen)
        hidden = torch.randn(POSITIONS, DIM, generator=gen)
        first = int(torch.randint(3, VOCAB, (1,), generator=gen))

        model = build_model(seen)
        fc = seen["eagle_linear"]
        # Pairs `(h[j], t[j + 1])` for the committed positions, then the chain:
        # step 1 pairs the last hidden row with the pending token, every later
        # step the previous step's normed output with the token it drafted.
        pair_tokens = list(tokens[1:].tolist()) + [first]
        pair_hidden = [hidden[j] for j in range(POSITIONS)]
        steps, margins, inputs = [], [], []
        for step in range(CHAIN):
            emb = embd[torch.tensor(pair_tokens)]
            hid = torch.stack(pair_hidden)
            inputs.append((emb, hid))
            normed = drafter(model, fc, emb, hid)[-1]
            logits = lm_head @ normed
            top = logits.topk(2).values
            margins.append(float(top[0] - top[1]))
            steps.append({"logits": logits.tolist(), "normed": normed.tolist(),
                          "token": int(logits.argmax())})
            pair_tokens.append(steps[-1]["token"])
            pair_hidden.append(normed)
        if min(margins) < 0.05:
            continue

        controls = {}
        no_llama4 = build_model(seen, llama4_beta=0.0)
        for name, run in {
            "hidden_first": lambda e, h: drafter(model, fc, e, h, hidden_first=True),
            "no_llama4_scaling": lambda e, h: drafter(no_llama4, fc, e, h),
            "position_plus_one": lambda e, h: drafter(model, fc, e, h, shift=1),
        }.items():
            controls[name] = [(lm_head @ run(e, h)[-1]).tolist() for e, h in inputs]

        save_file(file, str(HERE / "consolidated.safetensors"), metadata={"format": "pt"})
        (HERE / "params.json").write_text(json.dumps(params(), indent=2) + "\n")
        reference = {
            "seed": seed, "positions": POSITIONS, "first": first,
            "tokens": tokens.tolist(), "hidden": hidden.tolist(),
            "embd": embd.tolist(), "lm_head": lm_head.tolist(),
            "steps": steps, "controls": controls,
        }
        (HERE / "reference.json").write_text(json.dumps(reference) + "\n")
        print("seed", seed, "margins", [round(m, 3) for m in margins],
              "chain", [s["token"] for s in steps])
        return
    raise SystemExit("no seed with distinct top-2 logits")


if __name__ == "__main__":
    main()
