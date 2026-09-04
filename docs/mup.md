# muP and Muon scaling

Let `d` be the trunk width and `d0 = 64` the base width used for the tests, and
`m = d / d0` the width multiplier.

| group | init variance | LR multiplier | forward multiplier |
|---|---|---|---|
| embedding | `1` | `1` | `1` |
| trunk / expert matrices (Muon) | `1 / d_in` | global Muon LR, update scaled by `0.2 * sqrt(max(d_out, d_in))` | `1` |
| mixer / FFN output projections | `1 / (d_in * 2 * N_blocks)` | as above | `1` |
| norms, biases, router, halt head | `0` / `1` | `1` | `1` |
| unembedding | `1 / d_in` | `1 / m` | `1 / m` |
| PKM keys | `1 / 256` | `1` | `1` |
| PKM values (sparse AdamW) | `1 / d` | `1` | `1` |

The coordinate check asserts that activation RMS stays `O(1)` across widths
64, 128, 256 and 512 at initialisation and after 50 steps.
