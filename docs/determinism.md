# Deterministic mode

Deterministic mode guarantees two properties (R9):

1. **Run-to-run bit identity** on the same hardware class.
2. **Batch invariance**: a sequence's output does not depend on which other
   sequences share its batch.

Kernels that had to change, and how:

- `mla_decode` reduces over a fixed number of 64-token page groups in ascending
  split index order, and combines partial softmax states with a fixed tree
  shape. The unit test `decode_is_split_invariant_and_matches_prefill` asserts
  the output is identical for split factors 1, 2, 3, 4 and 8.
- `mla_prefill` uses an online softmax whose result is tile-size independent;
  asserted by `prefill_is_tile_size_invariant`.
- `router_topk_grouped` breaks every tie toward the lower index, for groups and
  for experts; asserted by `ties_break_toward_lower_index`.
- Expert dispatch is ordered by `(expert id, token id)` rather than arrival.
- Sampling draws from Philox keyed by `(request seed, sampling stream, step)`,
  so it is independent of thread count and batch composition.
- The halting threshold `tau` is fixed at its configured value; the scheduled
  halting jitter used during training is disabled.
