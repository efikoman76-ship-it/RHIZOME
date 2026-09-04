//! The versioned model schema and its validation rules (SPEC.md §3).

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::toml_lite::{self, Document};

/// Schema version written into weight files and plan hashes.
pub const SCHEMA_VERSION: u32 = 1;

/// Which front end / back end pair the model uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrontEnd {
    /// 128K BPE vocabulary with byte fallback.
    Bpe,
    /// Byte-latent patcher, encoder and decoder.
    Byte,
    /// Both paths are trained and selectable at serve time.
    Both,
}

impl FrontEnd {
    /// Parse from the config spelling.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "bpe" => Ok(FrontEnd::Bpe),
            "byte" => Ok(FrontEnd::Byte),
            "both" => Ok(FrontEnd::Both),
            other => Err(Error::Invalid(format!("unknown front_end `{other}`"))),
        }
    }

    /// Config spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            FrontEnd::Bpe => "bpe",
            FrontEnd::Byte => "byte",
            FrontEnd::Both => "both",
        }
    }

    /// Whether the BPE embedding / unembedding matrices exist.
    #[must_use]
    pub const fn has_bpe(self) -> bool {
        matches!(self, FrontEnd::Bpe | FrontEnd::Both)
    }

    /// Whether the byte encoder / decoder stack exists.
    #[must_use]
    pub const fn has_byte(self) -> bool {
        matches!(self, FrontEnd::Byte | FrontEnd::Both)
    }
}

/// Key/value source read by Think Core block C1 (SPEC.md §3.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadSource {
    /// Core-input cache of tokens `<= t` (default).
    Input,
    /// Previous-iteration latents (ablation only).
    PrevIter,
}

impl ReadSource {
    /// Parse from the config spelling.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "input" => Ok(ReadSource::Input),
            "prev_iter" => Ok(ReadSource::PrevIter),
            other => Err(Error::Invalid(format!("unknown core_read_source `{other}`"))),
        }
    }

    /// Config spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            ReadSource::Input => "input",
            ReadSource::PrevIter => "prev_iter",
        }
    }
}

/// A complete, validated model schema.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// Human readable configuration name (`test-s`, `edge`, ...).
    pub name: String,
    /// Schema version.
    pub schema_version: u32,
    /// Residual stream width.
    pub d_model: usize,
    /// Number of trunk strata (each stratum = 5 Local + 1 Global block).
    pub strata: usize,
    /// Think Core is inserted after this stratum (1-based).
    pub core_after_stratum: usize,
    /// Maximum Think Core iterations.
    pub l_max: usize,
    /// Attention / mixer head count.
    pub heads: usize,
    /// Per-head width.
    pub head_dim: usize,
    /// Routed expert count per MoE FFN.
    pub experts_routed: usize,
    /// Shared expert count per MoE FFN.
    pub experts_shared: usize,
    /// Top-k routed experts per token.
    pub top_k: usize,
    /// Expert SwiGLU hidden width.
    pub expert_hidden: usize,
    /// Router group count for group-limited routing.
    pub expert_groups: usize,
    /// 1-based block indices whose routed experts are replaced by PKM.
    pub pkm_blocks: Vec<usize>,
    /// Number of PKM value slots.
    pub pkm_slots: usize,
    /// Front end selection.
    pub front_end: FrontEnd,
    /// BPE vocabulary size.
    pub vocab: usize,
    /// Byte-path local width.
    pub d_local: usize,
    /// Think Core read source.
    pub core_read_source: ReadSource,
    /// MLA latent width.
    pub kv_lora: usize,
    /// Decoupled RoPE width.
    pub rope_dim: usize,
    /// Number of multi-token-prediction heads.
    pub mtp_heads: usize,
    /// Deployment weight format for experts and mixer projections.
    pub deploy_dtype: DType,
}

/// Blocks per stratum: 5 Local + 1 Global (SPEC.md §3.1).
pub const BLOCKS_PER_STRATUM: usize = 6;

impl ModelConfig {
    /// Total trunk blocks (excluding the Think Core).
    #[must_use]
    pub const fn trunk_blocks(&self) -> usize {
        self.strata * BLOCKS_PER_STRATUM
    }

    /// Total blocks used for depth-scaled init: trunk + 3 Core blocks.
    #[must_use]
    pub const fn depth_scale_blocks(&self) -> usize {
        self.trunk_blocks() + 3
    }

    /// Number of Local (Gated Delta) blocks in the trunk.
    #[must_use]
    pub const fn local_blocks(&self) -> usize {
        self.strata * 5
    }

    /// Number of Global (MLA) blocks in the trunk.
    #[must_use]
    pub const fn global_blocks(&self) -> usize {
        self.strata
    }

    /// Whether a 1-based trunk block index is a PKM block.
    #[must_use]
    pub fn is_pkm_block(&self, block_1based: usize) -> bool {
        self.pkm_blocks.contains(&block_1based)
    }

    /// Validate all cross-field invariants of the schema.
    ///
    /// # Errors
    /// Returns [`Error::Invalid`] describing the first violated rule.
    pub fn validate(&self) -> Result<()> {
        let inv = |m: String| Err(Error::Invalid(m));
        if self.schema_version != SCHEMA_VERSION {
            return inv(format!(
                "schema version {} != supported {SCHEMA_VERSION}",
                self.schema_version
            ));
        }
        if self.d_model == 0 || self.d_model % 2 != 0 {
            return inv("d_model must be positive and even".into());
        }
        if self.heads == 0 || self.head_dim == 0 {
            return inv("heads and head_dim must be positive".into());
        }
        if self.strata == 0 {
            return inv("strata must be >= 1".into());
        }
        if self.core_after_stratum == 0 || self.core_after_stratum > self.strata {
            return inv(format!(
                "core_after_stratum {} must be in 1..={}",
                self.core_after_stratum, self.strata
            ));
        }
        if self.l_max == 0 {
            return inv("l_max must be >= 1".into());
        }
        if self.top_k == 0 || self.top_k > self.experts_routed {
            return inv(format!(
                "top_k {} must be in 1..={}",
                self.top_k, self.experts_routed
            ));
        }
        if self.expert_groups == 0 || self.experts_routed % self.expert_groups != 0 {
            return inv("experts_routed must be a positive multiple of expert_groups".into());
        }
        // Group-limited routing takes the top-2 groups, so k must fit there.
        let per_group = self.experts_routed / self.expert_groups;
        let reachable = per_group * self.expert_groups.min(2);
        if self.top_k > reachable {
            return inv(format!(
                "top_k {} exceeds the {reachable} experts reachable in the top-2 groups",
                self.top_k
            ));
        }
        if self.expert_groups >= 2 && per_group < 3 && self.experts_routed > 4 {
            return inv("group scoring needs at least 3 experts per group".into());
        }
        for &b in &self.pkm_blocks {
            if b == 0 || b > self.trunk_blocks() {
                return inv(format!(
                    "pkm block {b} outside 1..={}",
                    self.trunk_blocks()
                ));
            }
        }
        if !self.pkm_blocks.is_empty() && self.pkm_slots == 0 {
            return inv("pkm_slots must be positive when pkm_blocks is non-empty".into());
        }
        if self.front_end.has_bpe() && self.vocab < 256 {
            return inv("vocab must cover at least the 256 byte fallbacks".into());
        }
        if self.front_end.has_byte() && self.d_local == 0 {
            return inv("d_local must be positive on the byte path".into());
        }
        if self.rope_dim >= self.kv_lora {
            return inv("rope_dim must be smaller than kv_lora".into());
        }
        Ok(())
    }

    /// Load and validate a configuration from TOML text.
    ///
    /// # Errors
    /// Parse, missing-key, type and validation errors are all reported.
    pub fn from_toml(text: &str) -> Result<Self> {
        let doc = toml_lite::parse(text)?;
        Self::from_document(&doc)
    }

    /// Build a configuration from an already parsed document.
    ///
    /// # Errors
    /// See [`ModelConfig::from_toml`].
    pub fn from_document(doc: &Document) -> Result<Self> {
        let g_usize = |k: &str| -> Result<usize> { doc.get(k)?.as_usize(k) };
        let opt_usize = |k: &str, d: usize| -> Result<usize> {
            match doc.try_get(k) {
                Some(v) => v.as_usize(k),
                None => Ok(d),
            }
        };
        let cfg = ModelConfig {
            name: doc.get("name")?.as_str("name")?.to_string(),
            schema_version: u32::try_from(opt_usize("schema_version", 1)?).map_err(|_| {
                Error::BadType {
                    key: "schema_version".into(),
                    expected: "u32",
                }
            })?,
            d_model: g_usize("model.d_model")?,
            strata: g_usize("model.strata")?,
            core_after_stratum: g_usize("model.core_after_stratum")?,
            l_max: g_usize("model.l_max")?,
            heads: g_usize("model.heads")?,
            head_dim: opt_usize("model.head_dim", 128)?,
            experts_routed: g_usize("moe.experts_routed")?,
            experts_shared: g_usize("moe.experts_shared")?,
            top_k: g_usize("moe.top_k")?,
            expert_hidden: g_usize("moe.expert_hidden")?,
            expert_groups: g_usize("moe.groups")?,
            pkm_blocks: match doc.try_get("pkm.blocks") {
                Some(v) => v.as_usize_array("pkm.blocks")?,
                None => Vec::new(),
            },
            pkm_slots: opt_usize("pkm.slots", 1_048_576)?,
            front_end: FrontEnd::parse(doc.get("frontend.kind")?.as_str("frontend.kind")?)?,
            vocab: opt_usize("frontend.vocab", 128_000)?,
            d_local: opt_usize("frontend.d_local", 0)?,
            core_read_source: match doc.try_get("model.core_read_source") {
                Some(v) => ReadSource::parse(v.as_str("model.core_read_source")?)?,
                None => ReadSource::Input,
            },
            kv_lora: opt_usize("model.kv_lora", 512)?,
            rope_dim: opt_usize("model.rope_dim", 64)?,
            mtp_heads: opt_usize("model.mtp_heads", 2)?,
            deploy_dtype: match doc.try_get("deploy.dtype") {
                Some(v) => match v.as_str("deploy.dtype")? {
                    "f32" => DType::F32,
                    "bf16" => DType::BF16,
                    "mxfp4" => DType::MXFP4,
                    "mxint4" => DType::MXINT4,
                    other => {
                        return Err(Error::Invalid(format!("unknown deploy dtype `{other}`")))
                    }
                },
                None => DType::F32,
            },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// The built-in `test-s` configuration (SPEC.md §3.10).
    #[must_use]
    pub fn test_s() -> Self {
        ModelConfig {
            name: "test-s".into(),
            schema_version: SCHEMA_VERSION,
            d_model: 64,
            strata: 1,
            core_after_stratum: 1,
            l_max: 3,
            heads: 2,
            head_dim: 32,
            experts_routed: 4,
            experts_shared: 1,
            top_k: 2,
            expert_hidden: 128,
            expert_groups: 2,
            pkm_blocks: Vec::new(),
            pkm_slots: 0,
            front_end: FrontEnd::Bpe,
            vocab: 1024,
            d_local: 0,
            core_read_source: ReadSource::Input,
            kv_lora: 64,
            rope_dim: 16,
            mtp_heads: 2,
            deploy_dtype: DType::F32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s_is_valid() {
        ModelConfig::test_s().validate().expect("built-in valid");
    }

    #[test]
    fn rejects_core_after_last_stratum() {
        let mut c = ModelConfig::test_s();
        c.core_after_stratum = 4;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_topk_beyond_two_groups() {
        let mut c = ModelConfig::test_s();
        c.top_k = 4;
        assert!(c.validate().is_err());
    }

    #[test]
    fn round_trips_through_toml() {
        let text = r#"
name = "unit"
[model]
d_model = 128
strata = 2
core_after_stratum = 1
l_max = 4
heads = 4
head_dim = 32
kv_lora = 64
rope_dim = 16
[moe]
experts_routed = 8
experts_shared = 1
top_k = 2
expert_hidden = 256
groups = 2
[frontend]
kind = "bpe"
vocab = 1024
"#;
        let cfg = ModelConfig::from_toml(text).expect("parses");
        assert_eq!(cfg.trunk_blocks(), 12);
        assert_eq!(cfg.local_blocks(), 10);
        assert_eq!(cfg.global_blocks(), 2);
    }
}
