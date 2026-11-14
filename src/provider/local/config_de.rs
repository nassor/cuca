//! The hand-written `Deserialize` impls of the local configuration types, and
//! `Display` for [`LocalDevice`]: the parts of the KDL mapping a derive
//! cannot express. A value with a string grammar (a device, a quantization)
//! parses through its `FromStr`; a source is a string (a preset or a
//! repository reference) or a table.

use std::fmt;
use std::str::FromStr;

use serde::de::{Deserialize, Deserializer, Error as _};

use super::{DrafterSource, EngramQuant, GpuAdapter, LocalDevice, ModelSource, WeightQuant};
use crate::config::ConfigValue;
use crate::error::CucaError;

/// The text of a parse error, without `CucaError`'s `configuration error:`
/// prefix: the deserializer's own error already says where it happened.
fn reason(error: CucaError) -> String {
    match error {
        CucaError::Config(message) => message,
        other => other.to_string(),
    }
}

/// A string parsed with `T`'s `FromStr`.
fn parsed<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr<Err = CucaError>,
{
    let text = String::deserialize(deserializer)?;
    text.parse().map_err(|e| D::Error::custom(reason(e)))
}

impl<'de> Deserialize<'de> for LocalDevice {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        parsed(deserializer)
    }
}

impl<'de> Deserialize<'de> for WeightQuant {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        parsed(deserializer)
    }
}

impl<'de> Deserialize<'de> for EngramQuant {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        parsed(deserializer)
    }
}

/// The string `str::parse` reads back: `cpu`, `vulkan`, `vulkan:<n>`,
/// `vulkan:integrated:<n>` (the same for `metal`), `cuda:<n>`, `rocm:<n>`.
impl fmt::Display for LocalDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let adapter = |f: &mut fmt::Formatter<'_>, name: &str, adapter: &GpuAdapter| match adapter {
            GpuAdapter::Default => f.write_str(name),
            GpuAdapter::Discrete(n) => write!(f, "{name}:{n}"),
            GpuAdapter::Integrated(n) => write!(f, "{name}:integrated:{n}"),
        };
        match self {
            LocalDevice::Cpu => f.write_str("cpu"),
            LocalDevice::Vulkan(a) => adapter(f, "vulkan", a),
            LocalDevice::Metal(a) => adapter(f, "metal", a),
            LocalDevice::Cuda(n) => write!(f, "cuda:{n}"),
            LocalDevice::Rocm(n) => write!(f, "rocm:{n}"),
        }
    }
}

/// A [`ModelSource`] from a string (`str::parse`: a preset, `repo:revision:file`
/// or `repo:revision`) or a table (`file=`, `dir=`, `checkpoint { ... }`,
/// `hugging_face { ... }`). A string never names a path, so this form never
/// touches the filesystem.
pub(super) fn source<'de, D: Deserializer<'de>>(deserializer: D) -> Result<ModelSource, D::Error> {
    source_from_value(ConfigValue::deserialize(deserializer)?).map_err(D::Error::custom)
}

fn source_from_value(value: ConfigValue) -> Result<ModelSource, String> {
    match value {
        ConfigValue::String(text) => text.parse().map_err(reason),
        other => ModelSource::deserialize(other).map_err(|e| e.to_string()),
    }
}

/// `"in-trunk"` (ASCII case-insensitive) is [`DrafterSource::InTrunk`]; any
/// other value is a [`DrafterSource::File`] source in the forms
/// `LocalModelConfig::source` takes: a preset or repository string, or a
/// `file=`, `dir=`, `checkpoint` or `hugging_face` table.
impl<'de> Deserialize<'de> for DrafterSource {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = ConfigValue::deserialize(deserializer)?;
        if value
            .as_str()
            .is_some_and(|s| s.trim().eq_ignore_ascii_case("in-trunk"))
        {
            return Ok(DrafterSource::InTrunk);
        }
        source_from_value(value)
            .map(DrafterSource::File)
            .map_err(D::Error::custom)
    }
}
