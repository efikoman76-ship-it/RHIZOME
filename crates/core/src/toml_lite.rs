//! A minimal, dependency-free TOML subset parser.
//!
//! RHIZOME configuration files use a deliberately small subset of TOML:
//! top-level tables, integer / float / boolean / string values, and arrays of
//! integers or strings. Keeping the parser in-tree removes the last build
//! dependency, which is what makes fully hermetic CI possible (ADR 0001).

use crate::error::{Error, Result};
use std::collections::BTreeMap;

/// A parsed configuration value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Integer scalar.
    Int(i64),
    /// Floating point scalar.
    Float(f64),
    /// Boolean scalar.
    Bool(bool),
    /// String scalar.
    Str(String),
    /// Homogeneous array.
    Array(Vec<Value>),
}

impl Value {
    /// Read this value as an integer.
    pub fn as_int(&self, key: &str) -> Result<i64> {
        match self {
            Value::Int(i) => Ok(*i),
            _ => Err(Error::BadType {
                key: key.to_string(),
                expected: "integer",
            }),
        }
    }

    /// Read this value as an unsigned size.
    pub fn as_usize(&self, key: &str) -> Result<usize> {
        let v = self.as_int(key)?;
        usize::try_from(v).map_err(|_| Error::BadType {
            key: key.to_string(),
            expected: "non-negative integer",
        })
    }

    /// Read this value as a float (integers are promoted).
    pub fn as_float(&self, key: &str) -> Result<f64> {
        match self {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err(Error::BadType {
                key: key.to_string(),
                expected: "float",
            }),
        }
    }

    /// Read this value as a boolean.
    pub fn as_bool(&self, key: &str) -> Result<bool> {
        match self {
            Value::Bool(b) => Ok(*b),
            _ => Err(Error::BadType {
                key: key.to_string(),
                expected: "boolean",
            }),
        }
    }

    /// Read this value as a string slice.
    pub fn as_str(&self, key: &str) -> Result<&str> {
        match self {
            Value::Str(s) => Ok(s),
            _ => Err(Error::BadType {
                key: key.to_string(),
                expected: "string",
            }),
        }
    }

    /// Read this value as an array of unsigned sizes.
    pub fn as_usize_array(&self, key: &str) -> Result<Vec<usize>> {
        match self {
            Value::Array(items) => items.iter().map(|v| v.as_usize(key)).collect(),
            _ => Err(Error::BadType {
                key: key.to_string(),
                expected: "array of integers",
            }),
        }
    }
}

/// A flat document: keys are `table.key` (or just `key` at top level).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Document {
    entries: BTreeMap<String, Value>,
}

impl Document {
    /// Look up a key, returning an error when it is absent.
    pub fn get(&self, key: &str) -> Result<&Value> {
        self.entries
            .get(key)
            .ok_or_else(|| Error::MissingKey(key.to_string()))
    }

    /// Look up a key, returning `None` when it is absent.
    #[must_use]
    pub fn try_get(&self, key: &str) -> Option<&Value> {
        self.entries.get(key)
    }

    /// Iterate over every `(key, value)` pair in sorted key order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.entries.iter()
    }

    /// Insert a value, replacing any previous binding.
    pub fn insert(&mut self, key: impl Into<String>, value: Value) {
        self.entries.insert(key.into(), value);
    }
}

/// Parse a TOML-subset document.
///
/// # Errors
/// Returns [`Error::Parse`] on malformed input.
pub fn parse(input: &str) -> Result<Document> {
    let mut doc = Document::default();
    let mut prefix = String::new();
    for (idx, raw) in input.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        let lineno = idx + 1;
        if let Some(rest) = line.strip_prefix('[') {
            let name = rest.strip_suffix(']').ok_or_else(|| Error::Parse {
                line: lineno,
                reason: "unterminated table header".into(),
            })?;
            let name = name.trim();
            if name.is_empty() {
                return Err(Error::Parse {
                    line: lineno,
                    reason: "empty table name".into(),
                });
            }
            prefix = format!("{name}.");
            continue;
        }
        let (k, v) = line.split_once('=').ok_or_else(|| Error::Parse {
            line: lineno,
            reason: "expected `key = value`".into(),
        })?;
        let key = k.trim();
        if key.is_empty() {
            return Err(Error::Parse {
                line: lineno,
                reason: "empty key".into(),
            });
        }
        let value = parse_value(v.trim(), lineno)?;
        doc.insert(format!("{prefix}{key}"), value);
    }
    Ok(doc)
}

fn strip_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut in_string = false;
    for (i, b) in bytes.iter().enumerate() {
        match b {
            b'"' => in_string = !in_string,
            b'#' if !in_string => return &line[..i],
            _ => {}
        }
    }
    line
}

fn parse_value(text: &str, line: usize) -> Result<Value> {
    if text.is_empty() {
        return Err(Error::Parse {
            line,
            reason: "missing value".into(),
        });
    }
    if let Some(rest) = text.strip_prefix('[') {
        let inner = rest.strip_suffix(']').ok_or_else(|| Error::Parse {
            line,
            reason: "unterminated array".into(),
        })?;
        let inner = inner.trim();
        if inner.is_empty() {
            return Ok(Value::Array(Vec::new()));
        }
        let mut items = Vec::new();
        for part in split_top_level(inner) {
            items.push(parse_value(part.trim(), line)?);
        }
        return Ok(Value::Array(items));
    }
    if let Some(rest) = text.strip_prefix('"') {
        let inner = rest.strip_suffix('"').ok_or_else(|| Error::Parse {
            line,
            reason: "unterminated string".into(),
        })?;
        return Ok(Value::Str(inner.to_string()));
    }
    match text {
        "true" => return Ok(Value::Bool(true)),
        "false" => return Ok(Value::Bool(false)),
        _ => {}
    }
    let normalized: String = text.chars().filter(|c| *c != '_').collect();
    if let Ok(i) = normalized.parse::<i64>() {
        return Ok(Value::Int(i));
    }
    if let Ok(f) = normalized.parse::<f64>() {
        return Ok(Value::Float(f));
    }
    Err(Error::Parse {
        line,
        reason: format!("unrecognised value `{text}`"),
    })
}

fn split_top_level(input: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut start = 0usize;
    for (i, c) in input.char_indices() {
        match c {
            '"' => in_string = !in_string,
            '[' if !in_string => depth += 1,
            ']' if !in_string => depth = depth.saturating_sub(1),
            ',' if !in_string && depth == 0 => {
                out.push(&input[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = input[start..].trim();
    if !tail.is_empty() {
        out.push(&input[start..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scalars_tables_and_arrays() {
        let doc = parse(
            r#"
name = "test-s" # trailing comment
[model]
d_model = 64
tau = 0.5
byte_path = false
pkm_blocks = [12, 24, 36]
"#,
        )
        .expect("valid document");
        assert_eq!(doc.get("name").unwrap().as_str("name").unwrap(), "test-s");
        assert_eq!(
            doc.get("model.d_model").unwrap().as_usize("d").unwrap(),
            64
        );
        assert!((doc.get("model.tau").unwrap().as_float("t").unwrap() - 0.5).abs() < 1e-12);
        assert!(!doc.get("model.byte_path").unwrap().as_bool("b").unwrap());
        assert_eq!(
            doc.get("model.pkm_blocks")
                .unwrap()
                .as_usize_array("p")
                .unwrap(),
            vec![12, 24, 36]
        );
    }

    #[test]
    fn hash_inside_string_is_not_a_comment() {
        let doc = parse("s = \"a#b\"").unwrap();
        assert_eq!(doc.get("s").unwrap().as_str("s").unwrap(), "a#b");
    }

    #[test]
    fn reports_line_numbers() {
        let err = parse("a = 1\nb\n").unwrap_err();
        match err {
            Error::Parse { line, .. } => assert_eq!(line, 2),
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn empty_array_round_trips() {
        let doc = parse("x = []").unwrap();
        assert_eq!(doc.get("x").unwrap().as_usize_array("x").unwrap(), vec![]);
    }
}
