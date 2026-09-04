# Kernel math, adjoints and tolerances

Tolerance form: `|out - ref| <= atol + rtol * |ref|`, elementwise.

| op | adjoint | atol | rtol |
|---|---|---|---|
| add, mul, residual_add | exact linear rules | 1e-6 | 1e-5 |
| silu | `sigma(x) (1 + x (1 - sigma(x)))` | 1e-6 | 1e-5 |
| swiglu | `dsilu(a) b`, `silu(a)` | 1e-6 | 1e-5 |
| softmax | `y * (g - <y, g>)` | 1e-6 | 1e-5 |
| rmsnorm | `g s / rms - x <g s, x> / (n rms^3)` | 1e-6 | 1e-5 |
| linear | `g W`, `g^T x` | 1e-6 | 1e-5 |
| grouped_linear | per-group `linear` adjoint | 1e-6 | 1e-5 |
| rope_apply | rotation by `-theta` | 1e-6 | 1e-5 |
| softcap | `1 - tanh^2(x / cap)` | 1e-6 | 1e-5 |
| gated_delta_scan_chunked | reverse scan of the rank-1 update | 1e-6 | 1e-5 |
| mla_prefill / mla_decode | flash backward with recomputed logits | 1e-6 | 1e-5 |
| cross_entropy_zloss | `p - onehot + 2 z logZ p` | 1e-6 | 1e-5 |
| halt_head | `sigma(x) - target` | 1e-6 | 1e-5 |
| router_topk_grouped | non-differentiable selection; gates differentiable | 1e-6 | 1e-5 |
| pkm_topk_product | non-differentiable | n/a | n/a |
| sampling | non-differentiable | n/a | n/a |
| quantize_mx, quantize_fp8_block | straight-through | 1e-2 * RMS(ref) | 2e-2 |

bf16, fp8 and MX backends use `atol = 1e-2 * RMS(ref)` and `rtol = 2e-2`.
Gradient checks use f64 central differences with `h = 1e-5` and require a
relative error of at most `1e-6`, with inputs sampled away from kinks.
