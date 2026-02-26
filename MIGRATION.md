# SDK Migration Report — Vision Agent

> Modules extracted from [GetStream/Vision-Agents](https://github.com/GetStream/Vision-Agents) (Apache-2.0 License).

---

## License & Attribution

- **Upstream**: GetStream/Vision-Agents — Apache License 2.0
- **This Project**: MIT License
- **Compatibility**: Apache-2.0 → MIT is permitted (attribution required)
- **Attribution**: Preserved in `backend/THIRD_PARTY_LICENSES.md`

---

## Extracted Modules (22 Total)

| # | Module | Source Pattern | Purpose | Tests |
|---|--------|---------------|---------|-------|
| 1 | `llm_types.py` | SDK type system | ContentPart, NormalizedResponse, ToolSchema types | ✅ |
| 2 | `agent_core.py` | Agent orchestration | Tool-calling loop, multi-step reasoning | ✅ |
| 3 | `rag_engine.py` | RAG system | TF-IDF search over documents, zero external deps | ✅ |
| 4 | `instructions.py` | Instruction loader | Markdown instruction parser with @-mentions | ✅ |
| 5 | `warmup_cache.py` | Model pre-loading | Cache for YOLO/pose model weights | ✅ |
| 6 | `mcp_tools.py` | MCP integration | External tool server (Model Context Protocol) | ✅ |
| 7 | `profiling.py` | Performance | Timing decorators, latency tracking | ✅ |
| 8 | `turn_detection.py` | Voice VAD | Silence-based turn detection for voice input | ✅ |
| 9 | `transcript_buffer.py` | STT buffer | Accumulates partial transcripts into sentences | ✅ |
| 10 | `function_registry.py` | Tool registry | Decorator-based function registration for agents | ✅ |
| 11 | `event_bus.py` | Event system | Pub/sub event bus for inter-module communication | ✅ |
| 12 | `observability.py` | Metrics | Structured logging, counters, histograms | ✅ |
| 13 | `stt_engine.py` | Speech-to-text | Multi-provider STT engine (Gemini, Whisper) | ✅ |
| 14 | `stt_base.py` | STT base class | Abstract base for STT providers | ✅ |
| 15 | `edge_types.py` | Edge types | Type definitions for edge processing | ✅ |
| 16 | `config.py` | Configuration | Centralized config with env var loading | ✅ |
| 17 | `llm_provider.py` | LLM cascade | 7-tier provider chain with health tracking | ✅ |
| 18 | `llm_helpers.py` | LLM utilities | OpenAI client wrapper, prompt formatting | ✅ |
| 19 | `pose_engine.py` | Pose estimation | YOLOv8 pose model, joint angle calculation | ✅ |
| 20 | `security_cam.py` | Security | Threat detection, alert generation | ✅ |
| 21 | `crowd_monitor.py` | Crowd analysis | Density estimation, safety scoring | ✅ |
| 22 | `gaming_companion.py` | Gaming AI | Screenshot analysis, strategy advice | ✅ |

---

## API Changes & Adaptations

| Module | What Changed |
|--------|-------------|
| `agent_core.py` | Replaced WebRTC transport with HTTP chunking; added FastReply path |
| `rag_engine.py` | Used built-in TF-IDF instead of external vector DB dependency |
| `llm_provider.py` | Extended from 3 tiers to 7 tiers; added health tracking per-provider |
| `turn_detection.py` | Simplified from WebSocket-based to HTTP polling pattern |
| `function_registry.py` | Added `list_tools()` method for agent introspection |
| `event_bus.py` | Added async support and wildcard event matching |
| `observability.py` | Added histogram support and JSON export |

---

## Verification Results

```
Test Suite: test_deep_sdk.py
Tests Run:  50
Passed:     50
Failed:      0
Skipped:     0
Coverage:    All 22 modules imported and smoke-tested
```

---

## Files NOT Copied (and why)

| File/Dir | Reason |
|----------|--------|
| `examples/webrtc/` | WebRTC-specific, replaced with HTTP streaming |
| `plugins/video-filters/` | Browser-side WebGL filters, not applicable to backend |
| `docker/` | Custom Docker setup, we use our own `docker-compose.yml` |
| `tests/e2e/` | End-to-end tests tied to upstream infrastructure |

---

## Recommended Next Steps

1. **WebRTC Integration**: Re-integrate WebRTC transport for full-duplex real-time streaming
2. **Plugin Marketplace**: Expose `function_registry` as plugin API for community extensions
3. **MCP Server**: Deploy `mcp_tools.py` as standalone MCP server for external tool access
4. **Benchmarking**: Run systematic latency benchmarks across all 7 LLM tiers
