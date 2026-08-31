//! Canonical JSON encoding for `serde_json::Value` leaves.
//!
//! [`CanonicalValue`] serializes a [`serde_json::Value`] as compact JSON text
//! with every object's keys sorted recursively. Array order and scalar values
//! are preserved. The sorted order is emitted directly from the source value,
//! so no intermediate [`serde_json::Value`] tree is built.
//!
//! # Why the text intermediate
//!
//! `serde_json::Value` serializes untagged: a `Value` field encoded natively
//! into a binary format loses its type. Under postcard, `false`, `0`, `""`,
//! `[]`, and `{}` all become the single byte `0x00`, so distinct payloads
//! produce identical bytes. Encoding each leaf as length-prefixed JSON text
//! keeps it self-describing, which is what makes a postcard digest injective
//! and a postcard record decodable.
//!
//! # Why sorting is explicit
//!
//! `serde_json::Map` is `BTreeMap`-backed, and therefore already ordered,
//! unless `preserve_order` is enabled somewhere in the dependency graph. A
//! downstream crate can enable it through feature unification, so the order is
//! established here rather than inherited from a build detail.

use serde::{Serialize, Serializer};

/// A [`serde_json::Value`] that serializes as canonical JSON text.
///
/// Object keys are sorted at every depth; arrays keep their order. The output
/// is a single string value, so any format that can encode a string can carry
/// it.
pub(crate) struct CanonicalValue<'a>(pub(crate) &'a serde_json::Value);

impl Serialize for CanonicalValue<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde_json::to_string(&SortedKeys(self.0))
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Emits `value` with object keys sorted at every depth.
///
/// Separate from [`CanonicalValue`] because this shape writes structured JSON,
/// while [`CanonicalValue`] wraps the finished text in a string.
struct SortedKeys<'a>(&'a serde_json::Value);

impl Serialize for SortedKeys<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            serde_json::Value::Object(map) => {
                use serde::ser::SerializeMap;
                // Sorting borrowed keys avoids rebuilding the map: only the
                // pointer vector is allocated, never a second Value tree.
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort_unstable();
                let mut out = serializer.serialize_map(Some(keys.len()))?;
                for key in keys {
                    out.serialize_entry(key, &SortedKeys(&map[key]))?;
                }
                out.end()
            }
            serde_json::Value::Array(items) => {
                use serde::ser::SerializeSeq;
                let mut out = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    out.serialize_element(&SortedKeys(item))?;
                }
                out.end()
            }
            scalar => scalar.serialize(serializer),
        }
    }
}

/// `#[serde(with = ...)]` adapter carrying a [`serde_json::Value`] field as
/// canonical JSON text.
///
/// Round-trips: the encoded form is the canonical text, and decoding parses it
/// back. Key order inside the value is not preserved across the round trip,
/// which is what makes the encoding canonical.
#[cfg(feature = "plugin-session-log")]
pub(crate) mod value_as_canonical_json {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Encode `value` as canonical JSON text.
    pub(crate) fn serialize<S: Serializer>(
        value: &serde_json::Value,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        super::CanonicalValue(value).serialize(serializer)
    }

    /// Parse the canonical JSON text back into a [`serde_json::Value`].
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<serde_json::Value, D::Error> {
        let text = <&str>::deserialize(deserializer)?;
        serde_json::from_str(text).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical_text(value: &serde_json::Value) -> String {
        serde_json::to_value(CanonicalValue(value))
            .expect("canonical serialization should succeed")
            .as_str()
            .expect("canonical form is a JSON string")
            .to_string()
    }

    #[test]
    fn object_keys_are_sorted_at_every_depth() {
        let value = serde_json::json!({
            "z": { "m": 0, "b": [1, 2] },
            "a": 1
        });
        assert_eq!(canonical_text(&value), r#"{"a":1,"z":{"b":[1,2],"m":0}}"#);
    }

    #[test]
    fn key_insertion_order_does_not_change_the_text() {
        let a: serde_json::Value =
            serde_json::from_str(r#"{"a":1,"z":{"m":0,"b":[1,2]}}"#).expect("valid json");
        let b: serde_json::Value =
            serde_json::from_str(r#"{"z":{"b":[1,2],"m":0},"a":1}"#).expect("valid json");
        assert_eq!(canonical_text(&a), canonical_text(&b));
    }

    #[test]
    fn array_order_is_significant() {
        let a = serde_json::json!({ "a": [1, 2] });
        let b = serde_json::json!({ "a": [2, 1] });
        assert_ne!(canonical_text(&a), canonical_text(&b));
    }

    #[test]
    fn scalars_and_empty_containers_stay_distinct() {
        let cases = [
            serde_json::json!(null),
            serde_json::json!(false),
            serde_json::json!(true),
            serde_json::json!(0),
            serde_json::json!(1),
            serde_json::json!(""),
            serde_json::json!("0"),
            serde_json::json!([]),
            serde_json::json!({}),
        ];
        let mut seen = std::collections::HashSet::new();
        for case in &cases {
            assert!(
                seen.insert(canonical_text(case)),
                "canonical text for {case} collided"
            );
        }
        assert_eq!(seen.len(), cases.len());
    }

    #[cfg(feature = "plugin-session-log")]
    #[test]
    fn adapter_round_trips_through_postcard() {
        #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        struct Holder {
            #[serde(with = "super::value_as_canonical_json")]
            value: serde_json::Value,
        }

        let holder = Holder {
            value: serde_json::json!({ "z": [null, true, 1.5, "s", {}], "a": 1 }),
        };
        let bytes = postcard::to_stdvec(&holder).expect("encode");
        let back: Holder = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, holder);
    }
}
