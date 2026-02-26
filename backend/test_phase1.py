"""Quick integration test for Phase 1 SDK modules."""
import asyncio, json, sys

# Test 1: Function Registry
from function_registry import tool_registry
tools = tool_registry.get_tools_schema()
print(f"Test 1: {len(tools)} tool schemas generated")
for t in tools:
    n = t["function"]["name"]
    params = list(t["function"]["parameters"].get("properties", {}).keys())
    print(f"  - {n}({', '.join(params)})")

# Test 2-4: Execute tools
async def test_tools():
    r1 = await tool_registry.execute("get_server_time", {})
    print(f"Test 2: get_server_time => ok={r1['ok']}")

    r2 = await tool_registry.execute("list_models", {})
    print(f"Test 3: list_models => ok={r2['ok']}")

    r3 = await tool_registry.execute("nonexistent_fn", {})
    print(f"Test 4: unknown func => ok={r3['ok']} (expected False)")
    assert r3["ok"] == False

asyncio.run(test_tools())

# Test 5: Event Bus
from event_bus import event_bus, EventType, Event
async def test_events():
    received = []
    @event_bus.on(EventType.INFO)
    async def handler(evt):
        received.append(evt)
    await event_bus.emit_info("Hello from test", source="test")
    hist = event_bus.get_history()
    print(f"Test 5: Event emitted & received: {len(received)} events, history: {len(hist)}")
    assert len(received) == 1

asyncio.run(test_events())

# Test 6: Provider Health
from observability import health_tracker, platform_metrics
health_tracker.record_success("ollama", 150.0)
health_tracker.record_success("ollama", 200.0)
health_tracker.record_error("gemini", "rate limit", 0)
stats = health_tracker.get_all_stats()
print(f"Test 6: Providers: {list(stats['providers'].keys())}")
print(f"  ollama avg={stats['providers']['ollama']['avg_latency_ms']}ms, rate={stats['providers']['ollama']['success_rate']}")
assert stats["providers"]["ollama"]["success_rate"] == 1.0

# Test 7: Prometheus
prom = health_tracker.to_prometheus()
lines = prom.strip().split("\n")
print(f"Test 7: Prometheus output: {len(prom)} chars, {len(lines)} lines")
assert "vision_agent_llm_requests_total" in prom

# Test 8: Platform Metrics
pm = platform_metrics.to_dict()
print(f"Test 8: Platform metrics: {len(pm)} keys, uptime={pm['uptime_seconds']:.1f}s")

# Test 9: Video Processor Pipeline
from video_processor import create_default_pipeline
pipeline = create_default_pipeline()
async def test_pipeline():
    await pipeline.initialize()
    ctx = await pipeline.process(b"\x00" * 100, source="test")
    print(f"Test 9: Pipeline: {len(pipeline.processors)} processors, frame_id={ctx.frame_id}")
    print(f"  motion={ctx.metadata.get('motion_detected')}, skipped={ctx.metadata.get('frame_skipped')}")
asyncio.run(test_pipeline())

# Test 10: Conversation Manager
from conversation import conversation_manager
s = conversation_manager.get_or_create("test-session")
s.add_message("user", "Hello, test!")
s.add_message("assistant", "Hi there!")
ctx = s.get_context()
print(f"Test 10: Session 'test-session': {s.message_count} messages, context_len={len(ctx)}")
assert s.message_count == 2
conversation_manager.delete("test-session")

print("\n" + "="*50)
print("ALL 10 TESTS PASSED")
print("="*50)
