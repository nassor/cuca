//! Configuration as data: the serde helpers every configuration struct uses,
//! and (feature `config-kdl`) the KDL reader that fills those structs from a
//! file.
//!
//! cuca reads no environment variable for its settings. A library user builds
//! the configuration structs (`LocalModelConfig` and the types it holds) in
//! Rust; the `cuca` binary reads one KDL file whose keys are the same structs'
//! field names, so a setting is spelled the same in both. The environment is
//! read only for the `${VAR}` placeholders a KDL file writes itself (see
//! `load_kdl_str`, feature `config-kdl`).
//!
//! The helpers here are the field adapters those structs carry:
//! [`one_or_many`] for a list written once or several times, [`from_str`] for
//! a value with a string grammar (a device, a quantization), [`size`] for a
//! byte count and [`duration`] for a time span.

use std::fmt::Display;
use std::str::FromStr;
use std::time::Duration;

use serde::Deserialize as _;
use serde::de::{self, DeserializeOwned, Deserializer, Visitor};

#[cfg(feature = "config-kdl")]
mod kdl;
#[cfg(feature = "config-kdl")]
pub use self::kdl::{from_kdl_str, from_kdl_str_with_vars, load_kdl_file, load_kdl_str};

/// A parsed configuration value.
///
/// `serde_json::Value` is the tree rather than a bespoke enum because it
/// already carries a `serde::Deserializer` that tagged enums, `flatten` and
/// `deny_unknown_fields` all work against, plus the accessors (`get`,
/// `as_str`, `as_object`) a reader of an opaque section needs.
pub type ConfigValue = serde_json::Value;

/// Deserialize a field that may be written once or several times into a `Vec`.
///
/// In KDL, same-named sibling nodes collapse into an array only from the
/// second occurrence, and a node with several arguments is an array, so a
/// one-element list is a single value in the tree. Every list-valued field
/// carries this.
///
/// Goes through [`ConfigValue`] first rather than an untagged enum so a
/// `deny_unknown_fields` violation inside one entry still names the offending
/// key.
///
/// # Errors
///
/// Returns the deserializer's own error when an entry does not match `T`.
pub fn one_or_many<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    use serde::de::Error as _;

    match ConfigValue::deserialize(deserializer)? {
        ConfigValue::Array(items) => items
            .into_iter()
            .map(|item| T::deserialize(item).map_err(D::Error::custom))
            .collect(),
        single => T::deserialize(single)
            .map(|value| vec![value])
            .map_err(D::Error::custom),
    }
}

/// Deserialize a string through `T`'s [`FromStr`].
///
/// # Errors
///
/// Returns the parse error's text when the string is not a `T`, and a type
/// error when the value is not a string.
pub fn from_str<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr,
    T::Err: Display,
{
    let text = String::deserialize(deserializer)?;
    text.parse().map_err(de::Error::custom)
}

/// Deserialize a byte count: an integer, or a string of digits with an
/// optional `k`, `m` or `g` suffix (powers of 1024, ASCII case-insensitive),
/// so `"4g"` is 4,294,967,296.
///
/// # Errors
///
/// Rejects a negative integer, any other string, and a count above
/// `u64::MAX`.
pub fn size<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
    struct SizeVisitor;

    impl Visitor<'_> for SizeVisitor {
        type Value = u64;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(SIZE_EXPECTED)
        }

        fn visit_u64<E: de::Error>(self, v: u64) -> Result<u64, E> {
            Ok(v)
        }

        fn visit_i64<E: de::Error>(self, v: i64) -> Result<u64, E> {
            u64::try_from(v).map_err(|_| E::custom(format!("{SIZE_EXPECTED}, got {v}")))
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<u64, E> {
            parse_size(v).ok_or_else(|| E::custom(format!("{SIZE_EXPECTED}, got `{v}`")))
        }
    }

    deserializer.deserialize_any(SizeVisitor)
}

const SIZE_EXPECTED: &str = "expected a byte count such as `4294967296`, `512m` or `4g`";

/// A byte count with an optional `k`, `m` or `g` suffix (powers of 1024,
/// ASCII case-insensitive).
fn parse_size(text: &str) -> Option<u64> {
    let text = text.trim();
    let (digits, shift) = match text.as_bytes().last()?.to_ascii_lowercase() {
        b'k' => (&text[..text.len() - 1], 10),
        b'm' => (&text[..text.len() - 1], 20),
        b'g' => (&text[..text.len() - 1], 30),
        _ => (text, 0),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok()?.checked_mul(1u64 << shift)
}

/// Deserialize a duration: a string of digits followed by `ms`, `s`, `m` or
/// `h` (`"500ms"`, `"60s"`, `"5m"`, `"1h"`). A bare integer is an error, since
/// its unit would be a guess.
///
/// # Errors
///
/// Rejects anything else, including a bare number.
pub fn duration<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
    struct DurationVisitor;

    impl Visitor<'_> for DurationVisitor {
        type Value = Duration;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(DURATION_EXPECTED)
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Duration, E> {
            parse_duration(v).ok_or_else(|| E::custom(format!("{DURATION_EXPECTED}, got `{v}`")))
        }
    }

    deserializer.deserialize_str(DurationVisitor)
}

const DURATION_EXPECTED: &str = "expected a duration such as `500ms`, `60s`, `5m` or `1h`";

/// `<digits>ms`, `<digits>s`, `<digits>m` or `<digits>h`.
fn parse_duration(text: &str) -> Option<Duration> {
    let text = text.trim();
    let split = text.find(|c: char| !c.is_ascii_digit())?;
    let (digits, unit) = text.split_at(split);
    if digits.is_empty() {
        return None;
    }
    let n: u64 = digits.parse().ok()?;
    match unit {
        "ms" => Some(Duration::from_millis(n)),
        "s" => Some(Duration::from_secs(n)),
        "m" => n.checked_mul(60).map(Duration::from_secs),
        "h" => n.checked_mul(3600).map(Duration::from_secs),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Debug, Deserialize)]
    struct Sized {
        #[serde(deserialize_with = "size")]
        bytes: u64,
    }

    #[derive(Debug, Deserialize)]
    struct Timed {
        #[serde(deserialize_with = "duration")]
        every: Duration,
    }

    fn sized(value: ConfigValue) -> Result<u64, String> {
        Sized::deserialize(json!({ "bytes": value }))
            .map(|s| s.bytes)
            .map_err(|e| e.to_string())
    }

    fn timed(value: ConfigValue) -> Result<Duration, String> {
        Timed::deserialize(json!({ "every": value }))
            .map(|t| t.every)
            .map_err(|e| e.to_string())
    }

    #[test]
    fn a_size_is_an_integer_or_digits_with_a_binary_suffix() {
        assert_eq!(sized(json!(4096)), Ok(4096));
        assert_eq!(sized(json!("4096")), Ok(4096));
        assert_eq!(sized(json!("2k")), Ok(2048));
        assert_eq!(sized(json!("512M")), Ok(512 << 20));
        assert_eq!(sized(json!("4g")), Ok(4 << 30));
        for bad in [
            json!(-1),
            json!("g"),
            json!("1.5g"),
            json!("4t"),
            json!("-4g"),
            json!(""),
        ] {
            let err = sized(bad.clone()).unwrap_err();
            assert!(err.contains("expected a byte count"), "{bad}: {err}");
        }
        assert!(sized(json!("99999999999999g")).is_err());
    }

    #[test]
    fn a_duration_needs_its_unit() {
        assert_eq!(timed(json!("500ms")), Ok(Duration::from_millis(500)));
        assert_eq!(timed(json!("60s")), Ok(Duration::from_secs(60)));
        assert_eq!(timed(json!("5m")), Ok(Duration::from_secs(300)));
        assert_eq!(timed(json!("1h")), Ok(Duration::from_secs(3600)));
        for bad in [
            json!(60),
            json!("60"),
            json!("1d"),
            json!("ms"),
            json!("1.5s"),
        ] {
            let err = timed(bad.clone()).unwrap_err();
            assert!(err.contains("expected a duration"), "{bad}: {err}");
        }
    }

    #[test]
    fn one_or_many_takes_a_single_value_or_an_array() {
        #[derive(Debug, Deserialize)]
        struct List {
            #[serde(deserialize_with = "one_or_many")]
            items: Vec<u32>,
        }
        let one = List::deserialize(json!({ "items": 1 })).unwrap();
        assert_eq!(one.items, vec![1]);
        let two = List::deserialize(json!({ "items": [1, 2] })).unwrap();
        assert_eq!(two.items, vec![1, 2]);
    }

    #[test]
    fn from_str_reports_the_parse_error() {
        #[derive(Debug, Deserialize)]
        struct Number {
            #[serde(deserialize_with = "from_str")]
            n: u8,
        }
        assert_eq!(Number::deserialize(json!({ "n": "7" })).unwrap().n, 7);
        let err = Number::deserialize(json!({ "n": "300" })).unwrap_err();
        assert!(err.to_string().contains("too large"), "{err}");
    }
}
