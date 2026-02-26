# Third-Party Licenses

## GetStream/Vision-Agents SDK

- **Repository**: https://github.com/GetStream/Vision-Agents
- **License**: Apache License 2.0
- **Commit**: f684ece (February 2026)
- **Usage**: Core patterns, type system, and module structure adapted for this platform

### Components Extracted

| Module | Source File | Adaptation |
|--------|-----------|------------|
| `llm_types.py` | `core/llm/llm_types.py` | Adapted to dataclasses (removed TypedDict dependency) |
| `agent_core.py` | `core/agents/agents.py` | HTTP-adapted (removed WebRTC transport) |
| `rag_engine.py` | `core/rag/rag.py` | Added TF-IDF implementation |
| `instructions.py` | `core/instructions.py` | Added instruction presets |
| `warmup_cache.py` | `core/warmup.py` | Direct port with load-time tracking |
| `mcp_tools.py` | `core/mcp/` | HTTP-adapted (removed stdio transport) |
| `profiling.py` | `core/profiling/` | Added histogram statistics |
| `turn_detection.py` | `core/turn_detection/` | Direct port |
| `transcript_buffer.py` | `core/agents/transcript_buffer.py` | Enhanced with speaker diarization |
| `function_registry.py` | `core/llm/function_registry.py` | Enhanced with MCP support |
| `event_bus.py` | `core/events/manager.py` | Enhanced with SSE streaming |
| `observability.py` | `core/observability/collector.py` | Enhanced with Prometheus output |

### Apache License 2.0

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
