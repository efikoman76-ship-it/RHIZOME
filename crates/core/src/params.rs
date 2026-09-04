//! Parameter, FLOP and memory accounting behind `rhizome params`.
//!
//! The numbers produced here are normative: the spec-sync CI job asserts that
//! the table in SPEC.md equals `rhizome params --all --format=markdown`.

use crate::config::ModelConfig;
use crate::dtype::DType;

/// Accounting for one configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamReport {
    /// Configuration name.
    pub name: String,
    /// Total parameters, including all Think Core and byte-path weights.
    pub total: u64,
    /// Parameters used for a single token at Think Core depth `L = 1`.
    pub active_l1: u64,
    /// Additional active parameters per extra Think Core iteration.
    pub per_iteration: u64,
    /// Trunk (prelude + coda) parameters.
    pub trunk: u64,
    /// Think Core parameters (shared across iterations).
    pub core: u64,
    /// Embedding and unembedding parameters.
    pub embedding: u64,
    /// Product-key memory parameters.
    pub pkm: u64,
    /// Byte-path (patcher excluded) parameters.
    pub byte_path: u64,
    /// Deployment bytes for the whole model in its deploy dtype.
    pub deploy_bytes: u64,
    /// Per-token, per-layer latent KV cache values (MLA: kv_lora + rope_dim).
    pub kv_per_token_per_layer: u64,
}

fn mixer_local_params(c: &ModelConfig) -> u64 {
    let d = c.d_model as u64;
    let h = c.heads as u64;
    let dh = c.head_dim as u64;
    let inner = h * dh;
    // q, k, v, g projections + out projection.
    let proj = 4 * d * inner + inner * d;
    // causal depthwise conv (kernel 4) over q, k, v channels.
    let conv = 3 * inner * 4;
    // per-head scalars: alpha_logit, beta_logit weights and the learned a_h.
    let scalars = 2 * d * h + h;
    // per-head output RMSNorm scale + pre/sandwich norms.
    let norms = inner + 2 * d;
    proj + conv + scalars + norms
}

fn mixer_mla_params(c: &ModelConfig) -> u64 {
    let d = c.d_model as u64;
    let h = c.heads as u64;
    let dh = c.head_dim as u64;
    let lat = c.kv_lora as u64;
    let rope = c.rope_dim as u64;
    let q_lora = lat;
    // down-projections
    let down = d * lat + d * rope + d * q_lora;
    // up-projections: q_nope, q_rope from c_q; k_nope, v from c_kv
    let up = q_lora * h * dh + q_lora * h * rope + lat * h * dh + lat * h * dh;
    let out = h * dh * d;
    // norms: pre, sandwich, c_kv, c_q, per-head q_nope
    let norms = 2 * d + lat + q_lora + h * dh;
    down + up + out + norms
}

fn swiglu_params(d: u64, hidden: u64) -> u64 {
    // gate + up + down
    3 * d * hidden
}

fn moe_ffn_params(c: &ModelConfig, routed: bool) -> u64 {
    let d = c.d_model as u64;
    let he = c.expert_hidden as u64;
    let shared = c.experts_shared as u64 * swiglu_params(d, he);
    let routed_p = if routed {
        c.experts_routed as u64 * swiglu_params(d, he) + d * c.experts_routed as u64
    } else {
        0
    };
    shared + routed_p + 2 * d
}

fn pkm_block_params(c: &ModelConfig) -> u64 {
    let d = c.d_model as u64;
    let heads = 4u64;
    let q = 512u64;
    let keys = 2 * 1024 * 256;
    let queries = d * q * heads;
    let values = c.pkm_slots as u64 * d;
    let out = d * d;
    queries + keys + values + out
}

fn core_params(c: &ModelConfig) -> u64 {
    let d = c.d_model as u64;
    let adapter = 2 * d * d + 2 * d; // W_in over [r; h0] plus its RMSNorm
    let iter_emb = c.l_max as u64 * d;
    let budget_emb = c.l_max as u64 * d;
    let halt = d + d + 1;
    let c1 = mixer_mla_params(c) + moe_ffn_params(c, true);
    let c2 = mixer_local_params(c) + moe_ffn_params(c, true);
    let c3 = swiglu_params(d, 4 * d) + 2 * d;
    adapter + iter_emb + budget_emb + halt + c1 + c2 + c3
}

fn byte_path_params(c: &ModelConfig) -> u64 {
    if !c.front_end.has_byte() {
        return 0;
    }
    let dl = c.d_local as u64;
    let d = c.d_model as u64;
    let local_block = |width: u64| -> u64 {
        let inner = width;
        4 * width * inner + inner * width + 3 * inner * 4 + swiglu_params(width, 4 * width)
    };
    let encoder = 4 * local_block(dl) + 3 * 256_000 * dl + 256 * dl;
    let pool = dl + dl * dl * 3 + dl * d;
    let decoder = 6 * (local_block(dl) + 4 * dl * dl) + dl * 256;
    encoder + pool + decoder
}

fn mtp_params(c: &ModelConfig) -> u64 {
    let d = c.d_model as u64;
    let head = mixer_local_params(c) + swiglu_params(d, 4 * d) + 2 * d * d;
    c.mtp_heads as u64 * head
}

/// Compute the full accounting report for a configuration.
#[must_use]
pub fn report(c: &ModelConfig) -> ParamReport {
    let d = c.d_model as u64;
    let mut trunk = 0u64;
    let mut pkm = 0u64;
    for stratum in 0..c.strata {
        for within in 0..6 {
            let block_1based = stratum * 6 + within + 1;
            let is_global = within == 5;
            let mixer = if is_global {
                mixer_mla_params(c)
            } else {
                mixer_local_params(c)
            };
            let ffn = if c.is_pkm_block(block_1based) {
                let p = pkm_block_params(c);
                pkm += p;
                p + c.experts_shared as u64 * swiglu_params(d, c.expert_hidden as u64) + 2 * d
            } else {
                moe_ffn_params(c, true)
            };
            trunk += mixer + ffn;
        }
    }
    let core = core_params(c);
    let embedding = if c.front_end.has_bpe() {
        2 * c.vocab as u64 * d + d
    } else {
        0
    };
    let byte_path = byte_path_params(c);
    let mtp = mtp_params(c);
    let total = trunk + core + embedding + byte_path + mtp;

    // Active parameters: dense weights plus only the selected experts.
    let dense_expert_fraction = |routed: u64, shared: u64, k: u64| -> u64 {
        let he = c.expert_hidden as u64;
        (shared + k.min(routed)) * swiglu_params(d, he)
    };
    let k = c.top_k as u64;
    let mut active_trunk = 0u64;
    for stratum in 0..c.strata {
        for within in 0..6 {
            let block_1based = stratum * 6 + within + 1;
            let mixer = if within == 5 {
                mixer_mla_params(c)
            } else {
                mixer_local_params(c)
            };
            let ffn = if c.is_pkm_block(block_1based) {
                // top-32 of 1M value rows plus keys and query projections.
                let heads = 4u64;
                heads * 32 * d
                    + 2 * 1024 * 256
                    + d * 512 * heads
                    + d * d
                    + c.experts_shared as u64 * swiglu_params(d, c.expert_hidden as u64)
            } else {
                dense_expert_fraction(c.experts_routed as u64, c.experts_shared as u64, k)
                    + d * c.experts_routed as u64
            };
            active_trunk += mixer + ffn;
        }
    }
    let core_active = {
        let adapter = 2 * d * d + 2 * d;
        let halt = 2 * d + 1;
        let c1 = mixer_mla_params(c)
            + dense_expert_fraction(c.experts_routed as u64, c.experts_shared as u64, k);
        let c2 = mixer_local_params(c)
            + dense_expert_fraction(c.experts_routed as u64, c.experts_shared as u64, k);
        let c3 = swiglu_params(d, 4 * d);
        adapter + halt + c1 + c2 + c3
    };
    let active_l1 = active_trunk + core_active + embedding.min(2 * d * 2) + byte_path;

    let deploy_bytes = {
        let quantized = trunk + core + pkm.min(trunk);
        let dense = total.saturating_sub(quantized);
        (c.deploy_dtype.bytes_for(quantized as usize) + DType::BF16.bytes_for(dense as usize)) as u64
    };

    ParamReport {
        name: c.name.clone(),
        total,
        active_l1,
        per_iteration: core_active,
        trunk,
        core,
        embedding,
        pkm,
        byte_path,
        deploy_bytes,
        kv_per_token_per_layer: (c.kv_lora + c.rope_dim) as u64,
    }
}

/// Forward FLOPs for one token at Think Core depth `l` (2 per MAC).
#[must_use]
pub fn forward_flops(c: &ModelConfig, l: usize) -> u64 {
    let r = report(c);
    let per_iter = r.per_iteration;
    let base = r.active_l1.saturating_sub(per_iter);
    2 * (base + per_iter * l as u64)
}

/// Render one or more reports as a GitHub-flavoured markdown table.
#[must_use]
pub fn markdown_table(reports: &[ParamReport]) -> String {
    let mut s = String::from("| config | total params | active (L=1) | per iteration | trunk | core | embedding | pkm | byte path | deploy bytes | kv/token/layer |\n");
    s.push_str("|---|---|---|---|---|---|---|---|---|---|---|\n");
    for r in reports {
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            r.name,
            r.total,
            r.active_l1,
            r.per_iteration,
            r.trunk,
            r.core,
            r.embedding,
            r.pkm,
            r.byte_path,
            r.deploy_bytes,
            r.kv_per_token_per_layer
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn totals_are_consistent() {
        let c = ModelConfig::test_s();
        let r = report(&c);
        assert!(r.total > 0);
        assert!(r.active_l1 <= r.total);
        assert!(r.per_iteration > 0 && r.per_iteration < r.total);
        assert_eq!(r.kv_per_token_per_layer, (c.kv_lora + c.rope_dim) as u64);
    }

    #[test]
    fn flops_grow_linearly_with_depth() {
        let c = ModelConfig::test_s();
        let f1 = forward_flops(&c, 1);
        let f2 = forward_flops(&c, 2);
        let f3 = forward_flops(&c, 3);
        assert_eq!(f2 - f1, f3 - f2);
    }

    #[test]
    fn wider_models_have_more_parameters() {
        let small = ModelConfig::test_s();
        let mut big = small.clone();
        big.d_model = 128;
        big.name = "wide".into();
        assert!(report(&big).total > report(&small).total);
    }

    #[test]
    fn markdown_has_one_row_per_report() {
        let table = markdown_table(&[report(&ModelConfig::test_s())]);
        assert_eq!(table.lines().count(), 3);
    }
}
