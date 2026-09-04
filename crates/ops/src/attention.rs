//! Multi-head Latent Attention reference kernels (SPEC.md §3.3).

use crate::elementwise::{softcap, softmax};

/// Per-token latent cache entry: `[c_kv (kv_lora) ; k_rope (rope_dim)]`.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentCache {
    /// Latent width.
    pub kv_lora: usize,
    /// Decoupled RoPE width.
    pub rope_dim: usize,
    /// Row-major entries, one row of `kv_lora + rope_dim` per token.
    pub rows: Vec<f64>,
    /// Segment id per token, used for block-diagonal masking.
    pub segment: Vec<usize>,
}

impl LatentCache {
    /// Create an empty cache.
    #[must_use]
    pub fn new(kv_lora: usize, rope_dim: usize) -> Self {
        LatentCache {
            kv_lora,
            rope_dim,
            rows: Vec::new(),
            segment: Vec::new(),
        }
    }

    /// Width of one cached row.
    #[must_use]
    pub const fn row_width(&self) -> usize {
        self.kv_lora + self.rope_dim
    }

    /// Number of cached tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.segment.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append one token, returning `false` on a width mismatch.
    pub fn push(&mut self, row: &[f64], segment: usize) -> bool {
        if row.len() != self.row_width() {
            return false;
        }
        self.rows.extend_from_slice(row);
        self.segment.push(segment);
        true
    }

    /// Number of 64-token pages currently occupied.
    #[must_use]
    pub fn pages(&self) -> usize {
        self.len().div_ceil(64)
    }

    /// Borrow one row.
    #[must_use]
    pub fn row(&self, i: usize) -> &[f64] {
        let w = self.row_width();
        &self.rows[i * w..(i + 1) * w]
    }
}

/// Attention configuration shared by the prefill and decode kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnConfig {
    /// Non-RoPE per-head width.
    pub head_dim: usize,
    /// RoPE per-head width.
    pub rope_dim: usize,
    /// Optional logit softcap; `0` disables it.
    pub softcap: f64,
}

impl AttnConfig {
    /// Softmax scale `1 / sqrt(head_dim + rope_dim)`.
    #[must_use]
    pub fn scale(&self) -> f64 {
        1.0 / ((self.head_dim + self.rope_dim) as f64).sqrt()
    }
}

/// Naive O(n²) attention over explicit keys and values.
///
/// `q` has `head_dim + rope_dim` elements; `keys` is `n x (head_dim +
/// rope_dim)`; `values` is `n x head_dim`.
#[must_use]
pub fn attention_reference(
    cfg: &AttnConfig,
    q: &[f64],
    keys: &[f64],
    values: &[f64],
    mask: &[bool],
) -> Vec<f64> {
    let kw = cfg.head_dim + cfg.rope_dim;
    let n = mask.len();
    let mut logits = vec![f64::NEG_INFINITY; n];
    for (i, logit) in logits.iter_mut().enumerate() {
        if !mask[i] {
            continue;
        }
        let mut acc = 0.0;
        for p in 0..kw {
            acc += q[p] * keys[i * kw + p];
        }
        *logit = softcap(acc * cfg.scale(), cfg.softcap);
    }
    let w = softmax(&logits);
    let mut out = vec![0.0; cfg.head_dim];
    for i in 0..n {
        if !mask[i] {
            continue;
        }
        for (d, o) in out.iter_mut().enumerate() {
            *o += w[i] * values[i * cfg.head_dim + d];
        }
    }
    out
}

/// Up-project a latent cache row into `(key, value)` for one head.
#[must_use]
pub fn mla_up_project(
    cache_row: &[f64],
    kv_lora: usize,
    w_uk: &[f64],
    w_uv: &[f64],
    head_dim: usize,
) -> (Vec<f64>, Vec<f64>) {
    let c = &cache_row[..kv_lora];
    let rope = &cache_row[kv_lora..];
    let mut k = vec![0.0; head_dim + rope.len()];
    for (d, slot) in k.iter_mut().take(head_dim).enumerate() {
        let mut acc = 0.0;
        for (j, cv) in c.iter().enumerate() {
            acc += w_uk[d * kv_lora + j] * cv;
        }
        *slot = acc;
    }
    k[head_dim..].copy_from_slice(rope);
    let mut v = vec![0.0; head_dim];
    for (d, slot) in v.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (j, cv) in c.iter().enumerate() {
            acc += w_uv[d * kv_lora + j] * cv;
        }
        *slot = acc;
    }
    (k, v)
}

/// Flash-style tiled prefill over the latent cache (causal, per-segment).
///
/// Returns one `head_dim` output row per query position.
#[must_use]
pub fn mla_prefill(
    cfg: &AttnConfig,
    cache: &LatentCache,
    queries: &[f64],
    query_segment: &[usize],
    w_uk: &[f64],
    w_uv: &[f64],
    tile: usize,
) -> Vec<f64> {
    let qw = cfg.head_dim + cfg.rope_dim;
    let tile = tile.max(1);
    let mut out = Vec::with_capacity(query_segment.len() * cfg.head_dim);
    for (t, &seg) in query_segment.iter().enumerate() {
        let q = &queries[t * qw..(t + 1) * qw];
        // Online softmax across tiles: running max, sum and accumulator.
        let mut m = f64::NEG_INFINITY;
        let mut l = 0.0;
        let mut acc = vec![0.0; cfg.head_dim];
        let limit = (t + 1).min(cache.len());
        let mut start = 0usize;
        while start < limit {
            let end = (start + tile).min(limit);
            for i in start..end {
                if cache.segment[i] != seg {
                    continue;
                }
                let (k, v) = mla_up_project(cache.row(i), cache.kv_lora, w_uk, w_uv, cfg.head_dim);
                let mut dot = 0.0;
                for p in 0..qw {
                    dot += q[p] * k[p];
                }
                let logit = softcap(dot * cfg.scale(), cfg.softcap);
                let new_m = m.max(logit);
                let corr = if m.is_finite() {
                    (m - new_m).exp()
                } else {
                    0.0
                };
                let w = (logit - new_m).exp();
                l = l * corr + w;
                for d in 0..cfg.head_dim {
                    acc[d] = acc[d] * corr + w * v[d];
                }
                m = new_m;
            }
            start = end;
        }
        if l > 0.0 {
            for a in &mut acc {
                *a /= l;
            }
        }
        out.extend(acc);
    }
    out
}

/// Paged, split-K decode with the absorbed formulation.
///
/// `split` is the number of page groups reduced in a fixed order, which makes
/// the kernel batch-invariant (SPEC.md §6.8).
#[must_use]
pub fn mla_decode(
    cfg: &AttnConfig,
    cache: &LatentCache,
    query: &[f64],
    segment: usize,
    w_uk: &[f64],
    w_uv: &[f64],
    split: usize,
) -> Vec<f64> {
    let qw = cfg.head_dim + cfg.rope_dim;
    let n = cache.len();
    let split = split.max(1);
    let page = 64usize;
    let pages = n.div_ceil(page).max(1);
    let per_split = pages.div_ceil(split);
    // Partial (max, sum, acc) per split, combined in a fixed index order.
    let mut partials: Vec<(f64, f64, Vec<f64>)> = Vec::with_capacity(split);
    for s in 0..split {
        let p0 = s * per_split;
        let start = p0 * page;
        let end = ((p0 + per_split) * page).min(n);
        let mut m = f64::NEG_INFINITY;
        let mut l = 0.0;
        let mut acc = vec![0.0; cfg.head_dim];
        for i in start..end.max(start) {
            if cache.segment[i] != segment {
                continue;
            }
            let (k, v) = mla_up_project(cache.row(i), cache.kv_lora, w_uk, w_uv, cfg.head_dim);
            let mut dot = 0.0;
            for p in 0..qw {
                dot += query[p] * k[p];
            }
            let logit = softcap(dot * cfg.scale(), cfg.softcap);
            let new_m = m.max(logit);
            let corr = if m.is_finite() {
                (m - new_m).exp()
            } else {
                0.0
            };
            let w = (logit - new_m).exp();
            l = l * corr + w;
            for d in 0..cfg.head_dim {
                acc[d] = acc[d] * corr + w * v[d];
            }
            m = new_m;
        }
        partials.push((m, l, acc));
    }
    let gmax = partials
        .iter()
        .filter(|p| p.1 > 0.0)
        .fold(f64::NEG_INFINITY, |a, p| a.max(p.0));
    if !gmax.is_finite() {
        return vec![0.0; cfg.head_dim];
    }
    let mut total = 0.0;
    let mut out = vec![0.0; cfg.head_dim];
    for (m, l, acc) in &partials {
        if *l <= 0.0 {
            continue;
        }
        let corr = (m - gmax).exp();
        total += l * corr;
        for d in 0..cfg.head_dim {
            out[d] += acc[d] * corr;
        }
    }
    if total > 0.0 {
        for o in &mut out {
            *o /= total;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Philox;

    struct Fixture {
        cfg: AttnConfig,
        cache: LatentCache,
        w_uk: Vec<f64>,
        w_uv: Vec<f64>,
        queries: Vec<f64>,
        segs: Vec<usize>,
    }

    fn fixture(n: usize, segs: Vec<usize>) -> Fixture {
        let cfg = AttnConfig {
            head_dim: 8,
            rope_dim: 4,
            softcap: 0.0,
        };
        let kv_lora = 16;
        let mut r = Philox::new(99, 1);
        let mut cache = LatentCache::new(kv_lora, cfg.rope_dim);
        for i in 0..n {
            let row: Vec<f64> = (0..kv_lora + cfg.rope_dim)
                .map(|_| r.next_normal() * 0.3)
                .collect();
            assert!(cache.push(&row, segs[i]));
        }
        let w_uk: Vec<f64> = (0..cfg.head_dim * kv_lora)
            .map(|_| r.next_normal() * 0.2)
            .collect();
        let w_uv: Vec<f64> = (0..cfg.head_dim * kv_lora)
            .map(|_| r.next_normal() * 0.2)
            .collect();
        let queries: Vec<f64> = (0..n * (cfg.head_dim + cfg.rope_dim))
            .map(|_| r.next_normal() * 0.5)
            .collect();
        Fixture {
            cfg,
            cache,
            w_uk,
            w_uv,
            queries,
            segs,
        }
    }

    fn dense_reference(f: &Fixture, t: usize) -> Vec<f64> {
        let kw = f.cfg.head_dim + f.cfg.rope_dim;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        let mut mask = Vec::new();
        for i in 0..f.cache.len() {
            let (k, v) = mla_up_project(
                f.cache.row(i),
                f.cache.kv_lora,
                &f.w_uk,
                &f.w_uv,
                f.cfg.head_dim,
            );
            keys.extend(k);
            values.extend(v);
            mask.push(i <= t && f.cache.segment[i] == f.segs[t]);
        }
        let _ = kw;
        attention_reference(
            &f.cfg,
            &f.queries[t * kw..(t + 1) * kw],
            &keys,
            &values,
            &mask,
        )
    }

    #[test]
    fn prefill_matches_dense_reference() {
        let n = 20;
        let f = fixture(n, vec![0; n]);
        let got = mla_prefill(&f.cfg, &f.cache, &f.queries, &f.segs, &f.w_uk, &f.w_uv, 5);
        for t in 0..n {
            let want = dense_reference(&f, t);
            for d in 0..f.cfg.head_dim {
                assert!(
                    (got[t * f.cfg.head_dim + d] - want[d]).abs() < 1e-10,
                    "t={t} d={d}"
                );
            }
        }
    }

    #[test]
    fn prefill_is_tile_size_invariant() {
        let n = 17;
        let f = fixture(n, vec![0; n]);
        let a = mla_prefill(&f.cfg, &f.cache, &f.queries, &f.segs, &f.w_uk, &f.w_uv, 1);
        let b = mla_prefill(&f.cfg, &f.cache, &f.queries, &f.segs, &f.w_uk, &f.w_uv, 64);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn prefill_respects_segment_boundaries() {
        let segs = vec![0, 0, 0, 1, 1, 1];
        let f = fixture(6, segs.clone());
        let got = mla_prefill(&f.cfg, &f.cache, &f.queries, &f.segs, &f.w_uk, &f.w_uv, 2);
        let want = dense_reference(&f, 4);
        for d in 0..f.cfg.head_dim {
            assert!((got[4 * f.cfg.head_dim + d] - want[d]).abs() < 1e-10);
        }
    }

    #[test]
    fn decode_is_split_invariant_and_matches_prefill() {
        let n = 130;
        let f = fixture(n, vec![0; n]);
        let kw = f.cfg.head_dim + f.cfg.rope_dim;
        let q = &f.queries[(n - 1) * kw..n * kw];
        let base = mla_decode(&f.cfg, &f.cache, q, 0, &f.w_uk, &f.w_uv, 1);
        for split in [2usize, 3, 4, 8] {
            let got = mla_decode(&f.cfg, &f.cache, q, 0, &f.w_uk, &f.w_uv, split);
            for d in 0..f.cfg.head_dim {
                assert!((got[d] - base[d]).abs() < 1e-12, "split {split}");
            }
        }
        let pre = mla_prefill(&f.cfg, &f.cache, &f.queries, &f.segs, &f.w_uk, &f.w_uv, 32);
        for d in 0..f.cfg.head_dim {
            assert!((base[d] - pre[(n - 1) * f.cfg.head_dim + d]).abs() < 1e-10);
        }
    }

    #[test]
    fn cache_pages_are_64_tokens() {
        let f = fixture(65, vec![0; 65]);
        assert_eq!(f.cache.pages(), 2);
        assert_eq!(f.cache.row_width(), 20);
    }

    #[test]
    fn softcap_bounds_logits() {
        let cfg = AttnConfig {
            head_dim: 2,
            rope_dim: 0,
            softcap: 1.0,
        };
        let keys = [100.0, 0.0, 0.0, 1.0];
        let values = [1.0, 0.0, 0.0, 1.0];
        let out = attention_reference(&cfg, &[100.0, 0.0], &keys, &values, &[true, true]);
        // With a cap the far key still receives non-negligible mass.
        assert!(out[1] > 1e-3);
    }
}
