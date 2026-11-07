//! The model's sampling defaults and the precedence that resolves them.
//!
//! Highest first, for each of temperature, `top_k` and `top_p`:
//!
//! 1. the request (`UnifiedRequest::set_temperature`, `set_top_k`,
//!    `set_top_p`), applied per request by the scheduler;
//! 2. the value set in [`SamplingConfig`] (a field left `None` is not set);
//! 3. the model file's own default: the GGUF keys `general.sampling.temp`,
//!    `general.sampling.top_k` and `general.sampling.top_p`, which llama.cpp
//!    reads the same way (`common_params_sampling_apply_model_defaults`) and
//!    whose converter writes from `generation_config.json`
//!    (`temperature`, `top_k`, `top_p`; see `convert::config`);
//! 4. the built-in default ([`SamplingConfig::DEFAULT_TEMPERATURE`] and
//!    friends).
//!
//! A request `seed` wins over [`SamplingConfig::seed`]; the file carries no
//! seed. The other `general.sampling.*` keys (`min_p`, the repetition
//! penalty, XTC, mirostat, `sequence`) name samplers this engine does not
//! have and are not read; no model the crate is verified on sets `min_p`.

use std::collections::HashMap;

use candle_core::quantized::gguf_file::Value;

use super::SamplingConfig;

/// GGUF key of the default temperature.
const KEY_TEMP: &str = "general.sampling.temp";
/// GGUF key of the default `top_k`.
const KEY_TOP_K: &str = "general.sampling.top_k";
/// GGUF key of the default `top_p`.
const KEY_TOP_P: &str = "general.sampling.top_p";

/// The sampling defaults a model file carries; `None` where it names none
/// (or a value the engine cannot use).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct FileSampling {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_k: Option<usize>,
    pub(crate) top_p: Option<f32>,
}

/// A numeric metadata value as `f64`, whatever integer or float type the
/// converter wrote.
fn number(value: &Value) -> Option<f64> {
    Some(match value {
        Value::U8(n) => f64::from(*n),
        Value::I8(n) => f64::from(*n),
        Value::U16(n) => f64::from(*n),
        Value::I16(n) => f64::from(*n),
        Value::U32(n) => f64::from(*n),
        Value::I32(n) => f64::from(*n),
        Value::F32(n) => f64::from(*n),
        Value::F64(n) => *n,
        Value::U64(n) => *n as f64,
        Value::I64(n) => *n as f64,
        _ => return None,
    })
}

impl FileSampling {
    /// Read the defaults from a file's metadata. A value that is missing, not
    /// a number, or outside what [`SamplingConfig`] accepts (a negative or
    /// non-finite temperature, a `top_p` outside `(0, 1]`) is treated as
    /// absent, so an odd file never fails the load. A `top_k` of zero or less
    /// disables the filter, as in llama.cpp.
    pub(crate) fn from_metadata(md: &HashMap<String, Value>) -> Self {
        let get = |key: &str| md.get(key).and_then(number);
        Self {
            temperature: get(KEY_TEMP)
                .filter(|t| t.is_finite() && *t >= 0.0)
                .map(|t| t as f32),
            top_k: get(KEY_TOP_K)
                .filter(|k| k.is_finite())
                .map(|k| if k <= 0.0 { 0 } else { k as usize }),
            top_p: get(KEY_TOP_P)
                .filter(|p| *p > 0.0 && *p <= 1.0)
                .map(|p| p as f32),
        }
    }
}

/// The load-time sampling parameters after the precedence above: what a
/// request that sets none of them runs with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Sampling {
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) seed: Option<u64>,
}

impl Sampling {
    /// `config` over `file` over the built-in defaults.
    pub(crate) fn resolve(config: &SamplingConfig, file: &FileSampling) -> Self {
        Self {
            temperature: config
                .temperature
                .or(file.temperature)
                .unwrap_or(SamplingConfig::DEFAULT_TEMPERATURE),
            top_k: config
                .top_k
                .or(file.top_k)
                .unwrap_or(SamplingConfig::DEFAULT_TOP_K),
            top_p: config
                .top_p
                .or(file.top_p)
                .unwrap_or(SamplingConfig::DEFAULT_TOP_P),
            seed: config.seed,
        }
    }

    /// The resolved values as a fully set [`SamplingConfig`].
    pub(crate) fn config(self) -> SamplingConfig {
        SamplingConfig {
            temperature: Some(self.temperature),
            top_k: Some(self.top_k),
            top_p: Some(self.top_p),
            seed: self.seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn md(entries: &[(&str, Value)]) -> HashMap<String, Value> {
        entries
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn built_in_defaults_apply_without_a_file_or_config() {
        let s = Sampling::resolve(&SamplingConfig::default(), &FileSampling::default());
        assert_eq!(
            s,
            Sampling {
                temperature: 1.0,
                top_k: 64,
                top_p: 0.95,
                seed: None
            }
        );
    }

    #[test]
    fn file_defaults_beat_the_built_in_ones() {
        let file = FileSampling::from_metadata(&md(&[
            (KEY_TEMP, Value::F32(0.6)),
            (KEY_TOP_K, Value::I32(20)),
            (KEY_TOP_P, Value::F32(0.8)),
        ]));
        let s = Sampling::resolve(&SamplingConfig::default(), &file);
        assert_eq!((s.temperature, s.top_k, s.top_p), (0.6, 20, 0.8));
    }

    /// Each field resolves on its own: a configured value replaces only its
    /// own file default, and a seed never comes from the file.
    #[test]
    fn configured_values_beat_the_file_one_field_at_a_time() {
        let file = FileSampling {
            temperature: Some(0.6),
            top_k: Some(20),
            top_p: Some(0.8),
        };
        let config = SamplingConfig {
            temperature: Some(0.0),
            seed: Some(7),
            ..SamplingConfig::default()
        };
        let s = Sampling::resolve(&config, &file);
        assert_eq!(s.temperature, 0.0, "an explicit greedy temperature wins");
        assert_eq!((s.top_k, s.top_p), (20, 0.8), "the rest keeps the file's");
        assert_eq!(s.seed, Some(7));
        // A configured value equal to the built-in default is still explicit.
        let config = SamplingConfig {
            top_k: Some(SamplingConfig::DEFAULT_TOP_K),
            ..SamplingConfig::default()
        };
        assert_eq!(Sampling::resolve(&config, &file).top_k, 64);
    }

    #[test]
    fn the_file_is_read_whatever_number_type_it_holds() {
        let file = FileSampling::from_metadata(&md(&[
            (KEY_TEMP, Value::F64(0.7)),
            (KEY_TOP_K, Value::U32(40)),
            (KEY_TOP_P, Value::F32(1.0)),
        ]));
        assert_eq!(
            file,
            FileSampling {
                temperature: Some(0.7),
                top_k: Some(40),
                top_p: Some(1.0)
            }
        );
        // A non-positive top_k disables the filter, as in llama.cpp.
        let file = FileSampling::from_metadata(&md(&[(KEY_TOP_K, Value::I32(-1))]));
        assert_eq!(file.top_k, Some(0));
    }

    #[test]
    fn unusable_file_values_are_ignored_not_fatal() {
        let file = FileSampling::from_metadata(&md(&[
            (KEY_TEMP, Value::F32(-1.0)),
            (KEY_TOP_K, Value::String("many".into())),
            (KEY_TOP_P, Value::F32(0.0)),
        ]));
        assert_eq!(file, FileSampling::default());
        let file = FileSampling::from_metadata(&md(&[
            (KEY_TEMP, Value::F32(f32::NAN)),
            (KEY_TOP_P, Value::F32(1.5)),
        ]));
        assert_eq!(file, FileSampling::default());
    }

    #[test]
    fn the_resolved_values_report_as_a_set_config() {
        let s = Sampling::resolve(
            &SamplingConfig {
                seed: Some(3),
                ..SamplingConfig::default()
            },
            &FileSampling {
                temperature: Some(0.6),
                ..FileSampling::default()
            },
        );
        assert_eq!(
            s.config(),
            SamplingConfig {
                temperature: Some(0.6),
                top_k: Some(64),
                top_p: Some(0.95),
                seed: Some(3)
            }
        );
    }
}
