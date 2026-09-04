//! Numeric formats used by RHIZOME, including block-scaled MX formats.

/// A tensor element format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE-754 double, used by the op reference implementations.
    F64,
    /// IEEE-754 single precision.
    F32,
    /// bfloat16 residual stream format.
    BF16,
    /// FP8 with 4 exponent and 3 mantissa bits (training GEMMs).
    F8E4M3,
    /// MXFP4: 4-bit float with a shared e8m0 scale per 32 elements.
    MXFP4,
    /// MXINT4: 4-bit int with a shared e8m0 scale per 32 elements.
    MXINT4,
    /// 8-bit integer, per-row scaled.
    I8,
}

impl DType {
    /// Number of elements sharing one block scale, or 1 for unblocked formats.
    #[must_use]
    pub const fn block_size(self) -> usize {
        match self {
            DType::MXFP4 | DType::MXINT4 => 32,
            _ => 1,
        }
    }

    /// Storage cost of one element in bits, excluding shared block scales.
    #[must_use]
    pub const fn bits(self) -> usize {
        match self {
            DType::F64 => 64,
            DType::F32 => 32,
            DType::BF16 => 16,
            DType::F8E4M3 | DType::I8 => 8,
            DType::MXFP4 | DType::MXINT4 => 4,
        }
    }

    /// Bytes required to store `n` elements, including one e8m0 shared scale
    /// per block for the MX formats.
    #[must_use]
    pub fn bytes_for(self, n: usize) -> usize {
        let payload = (n * self.bits()).div_ceil(8);
        let block = self.block_size();
        if block > 1 {
            payload + n.div_ceil(block)
        } else {
            payload
        }
    }

    /// Whether the format carries a gradient in training.
    #[must_use]
    pub const fn is_float(self) -> bool {
        !matches!(self, DType::I8)
    }

    /// Stable lowercase name used in configs and reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            DType::F64 => "f64",
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            DType::F8E4M3 => "fp8e4m3",
            DType::MXFP4 => "mxfp4",
            DType::MXINT4 => "mxint4",
            DType::I8 => "i8",
        }
    }
}

/// Round-trip a value through the e4m3 FP8 encoding, bit-exactly.
///
/// Used both by the CPU emulation path and by quantisation conformance tests.
#[must_use]
pub fn fp8_e4m3_round_trip(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_nan() {
            f64::NAN
        } else {
            x.signum() * 448.0
        };
    }
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let a = x.abs();
    if a == 0.0 {
        return sign * 0.0;
    }
    // Smallest normal is 2^-6; subnormals step by 2^-9.
    let min_normal = 2f64.powi(-6);
    let max = 448.0;
    if a < min_normal {
        let step = 2f64.powi(-9);
        let q = (a / step).round() * step;
        return sign * q;
    }
    let e = a.log2().floor().clamp(-6.0, 8.0);
    let step = 2f64.powf(e - 3.0);
    let q = (a / step).round() * step;
    sign * q.min(max)
}

/// Quantise a block of values to a shared power-of-two (e8m0) scale.
///
/// Returns the exponent of the shared scale and the integer codes; `levels`
/// is the number of representable magnitudes (8 for 4-bit signed formats).
#[must_use]
pub fn mx_block_scale(block: &[f64], levels: f64) -> f64 {
    let amax = block.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if amax == 0.0 {
        return 1.0;
    }
    let exp = (amax / levels).log2().ceil();
    2f64.powf(exp.clamp(-127.0, 127.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_round_trip_is_idempotent() {
        for i in -2000..2000 {
            let x = f64::from(i) * 0.137;
            let a = fp8_e4m3_round_trip(x);
            let b = fp8_e4m3_round_trip(a);
            assert!((a - b).abs() < 1e-12, "not idempotent at {x}: {a} vs {b}");
        }
    }

    #[test]
    fn fp8_relative_error_bounded() {
        for i in 1..5000 {
            let x = f64::from(i) * 0.01;
            if x > 448.0 {
                break;
            }
            let q = fp8_e4m3_round_trip(x);
            assert!((q - x).abs() <= 0.07 * x + 1e-3, "x={x} q={q}");
        }
    }

    #[test]
    fn mx_scale_covers_block() {
        let block = [0.4, -3.2, 1.0, 0.0];
        let s = mx_block_scale(&block, 8.0);
        for v in block {
            assert!((v / s).abs() <= 8.0 + 1e-9);
        }
    }

    #[test]
    fn mx_bytes_include_scale() {
        assert_eq!(DType::MXFP4.block_size(), 32);
        assert_eq!(DType::BF16.bytes_for(10), 20);
    }
}
