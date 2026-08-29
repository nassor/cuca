⟦concern⟧ Design gap: specify whether a local cache HIT still runs plugin hooks (on_stream_chunk/on_response_complete). If a hit bypasses them, session-store trajectory logs, telemetry, and memory extraction silently miss that turn — breaking append-only session continuity; if it runs them, memory re-extracts entities on every replay. Pick and document one before the client seam gets built.
   
(Minor: your enumerated key material happens to equal the full UnifiedRequest today — hashing the canonical serialization of the whole struct is sturdier against future field additions.)

----------------

Memory size is always a concern. Consider implementing strategies such as data compression, efficient data structures, and selective caching to mitigate memory usage. Would be interesting to also put a memory cap, creating strategies to evict or offload less critical data when the cap is reached. Additionally, monitoring memory usage and setting up alerts when usage approaches the cap can help proactively manage memory resources. Finally, reviewing and optimizing memory usage patterns periodically can ensure that the system remains efficient over time.

[this should be a directive inside the @AGENTS.md]

----------------

Review plugins that can contribute with other plugins. I don't want the create to create a circular dependency across the plugin ecosystem, so carefully analyze the dependency graph and enforce rules to prevent cycles.

Other thing to consider is having features of a plugin that only work when certain other plugins are present, ensuring that optional dependencies are clearly documented and managed to avoid unexpected failures or missing functionality. I wonder if that is the case on the plugin-entity-extraction and plugin-memory plugin.

[this should be a directive inside the @AGENTS.md]