# 0003. Corrected test expectations found wrong by the first CI runs

- Status: accepted
- Date: 2026-09-04

## Context

R12 forbids weakening a test to go green unless an ADR shows the original
expectation was wrong. Four tests failed on the first full Windows run. Each
was examined before any change was made.

## Decision

Three test expectations were wrong and were corrected; one test exposed a real
implementation bug, which was fixed instead.

1. **`rng::normal_has_expected_moments` — implementation bug, not a test bug.**
   `Philox::next_f64` built its mantissa as `((hi << 21) ^ lo) >> 11`. `hi` is a
   full 32-bit word, so `hi << 21` occupies bits up to 52 *plus* carries above
   the 53-bit window, and the XOR with `lo` produced values whose top bits were
   correlated with `hi`. The empirical mean of the resulting Box–Muller normal
   was 4.15 instead of 0. Fixed by taking exactly 27 + 26 = 53 bits
   (`(hi >> 5) << 26 | (lo >> 6)`), which is uniform on `[0, 1)` by
   construction. The test threshold was **not** relaxed.

2. **`gradcheck::detects_an_incorrect_gradient`.** The test asserted
   `worst_index == 1`. The relative error is normalised per component and both
   components of a deliberately-zero analytic gradient are equally wrong up to
   floating-point noise, so which index wins is not a property of the checker.
   The assertion now checks the meaningful property: a wrong gradient produces a
   large relative error.

3. **`linalg::newton_schulz_produces_near_orthogonal_rows`.** The test asserted
   the Gram matrix is within 0.35 of the identity after 5 iterations. That is
   not what the Muon quintic guarantees: after Frobenius normalisation the
   singular values start well below one, and five iterations compress their
   spread rather than driving them to one. The test now asserts the properties
   Muon actually relies on — the diagonal spread of the Gram matrix strictly
   decreases, and off-diagonal mass stays below 25% of diagonal mass — plus a
   new test that the tall-matrix transpose path preserves shape and finiteness.

4. **`routing::balance_loss_is_lower_when_balanced`.** The fixture gave both the
   balanced and the skewed routing identical affinities `[0.25; 4]`. With equal
   `P_e`, `sum_e f_e P_e` is `0.25 * sum_e f_e`, and `sum_e f_e` is the same
   constant for any routing — so the two losses were provably equal and the
   assertion could never hold. The fixture now models a skewed router
   realistically: a router that overloads experts 0 and 1 also scores them
   highly, so its affinities are `[0.4, 0.4, 0.1, 0.1]`.

## Alternatives considered

- **Loosen the RNG moment thresholds.** Rejected outright: the generator was
  wrong, and R11 makes RNG quality load-bearing for reproducibility.
- **Increase the Newton–Schulz iteration count until the identity assertion
  passes.** Rejected: Muon deliberately uses exactly 5 iterations, and changing
  the optimizer to satisfy a mis-stated test inverts the dependency.

## Consequences

- `next_f64` and `stateless_uniform` produce different (correct) streams than
  the first commit. No checkpoints or golden files existed yet, so nothing is
  invalidated.
- The Newton–Schulz test now states a property that survives future
  coefficient tuning.
