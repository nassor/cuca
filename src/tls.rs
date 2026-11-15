//! TLS crypto provider for every reqwest client the crate builds.
//!
//! reqwest is compiled with `rustls-no-provider`, so a client can only be built
//! once the process has a default rustls `CryptoProvider`. [`ensure_crypto_provider`]
//! installs ring when none is installed and leaves an application's own choice
//! alone. This is the crate's one process-global side effect; an application that
//! builds its own `reqwest::Client` before any cuca client calls it first.

/// Installs ring as the process-default rustls provider unless one is set.
pub fn ensure_crypto_provider() {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        // A concurrent installer may win; either provider is valid.
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
}

/// `reqwest::Client::new()` with the provider in place.
#[cfg(any(
    test,
    feature = "provider-openai",
    feature = "provider-anthropic",
    feature = "provider-gemini",
    feature = "provider-deepseek",
    feature = "provider-llamacpp",
    feature = "provider-vllm",
    feature = "provider-lmstudio",
    feature = "plugin-web-search",
))]
pub(crate) fn http_client() -> reqwest::Client {
    ensure_crypto_provider();
    reqwest::Client::new()
}

/// `reqwest::Client::builder()` with the provider in place.
#[cfg(any(test, feature = "provider-local", feature = "plugin-web-search"))]
pub(crate) fn http_client_builder() -> reqwest::ClientBuilder {
    ensure_crypto_provider();
    reqwest::Client::builder()
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_client_builds_without_a_preinstalled_provider() {
        let _client = super::http_client();
        assert!(rustls::crypto::CryptoProvider::get_default().is_some());
    }
}
