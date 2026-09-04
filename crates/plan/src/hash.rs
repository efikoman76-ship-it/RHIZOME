//! Content hashing of plans (FNV-1a 64, stable across platforms).

use crate::graph::{Graph, Phase};

const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

/// Streaming FNV-1a hasher used for plan and schema hashes.
#[derive(Debug, Clone)]
pub struct Fnv1a(u64);

impl Default for Fnv1a {
    fn default() -> Self {
        Fnv1a(OFFSET)
    }
}

impl Fnv1a {
    /// Absorb raw bytes.
    pub fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(PRIME);
        }
    }

    /// Absorb a `usize` in little-endian form.
    pub fn write_usize(&mut self, v: usize) {
        self.write(&(v as u64).to_le_bytes());
    }

    /// Finish and return the digest.
    #[must_use]
    pub fn finish(&self) -> u64 {
        self.0
    }
}

/// Stable content hash of a plan for a given phase and batch bucket.
#[must_use]
pub fn plan_hash(graph: &Graph, phase: Phase, bucket: usize) -> u64 {
    let mut h = Fnv1a::default();
    h.write(phase.name().as_bytes());
    h.write_usize(bucket);
    for t in graph.tensors() {
        h.write(t.name.as_bytes());
        h.write_usize(t.elems);
        h.write(t.dtype.name().as_bytes());
        h.write(&[u8::from(t.is_param)]);
    }
    for n in graph.nodes() {
        h.write(n.op.name().as_bytes());
        for i in &n.inputs {
            h.write_usize(*i);
        }
        h.write_usize(n.output);
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OpKind;
    use rhizome_core::DType;

    fn g() -> Graph {
        let mut g = Graph::new();
        let x = g.tensor("x", 8, DType::F32, false);
        g.push(OpKind::Silu, &[x], "y", 8, DType::F32).expect("ok");
        g
    }

    #[test]
    fn hash_is_stable() {
        assert_eq!(
            plan_hash(&g(), Phase::Decode, 4),
            plan_hash(&g(), Phase::Decode, 4)
        );
    }

    #[test]
    fn hash_depends_on_phase_and_bucket() {
        assert_ne!(
            plan_hash(&g(), Phase::Decode, 4),
            plan_hash(&g(), Phase::Prefill, 4)
        );
        assert_ne!(
            plan_hash(&g(), Phase::Decode, 4),
            plan_hash(&g(), Phase::Decode, 8)
        );
    }

    #[test]
    fn hash_changes_with_graph_shape() {
        let mut other = g();
        let last = other.tensors().len() - 1;
        other
            .push(OpKind::Silu, &[last], "z", 8, DType::F32)
            .expect("ok");
        assert_ne!(
            plan_hash(&g(), Phase::Train, 1),
            plan_hash(&other, Phase::Train, 1)
        );
    }

    #[test]
    fn fnv_matches_known_vectors() {
        // Reference digests from the FNV-1a 64-bit specification. These pin
        // the constants: a mistyped prime silently changes every plan hash.
        for (input, want) in [
            ("", 0xcbf2_9ce4_8422_2325u64),
            ("a", 0xaf63_dc4c_8601_ec8c),
            ("foobar", 0x8594_4171_f739_67e8),
        ] {
            let mut h = Fnv1a::default();
            h.write(input.as_bytes());
            assert_eq!(h.finish(), want, "input {input:?}");
        }
    }
}
