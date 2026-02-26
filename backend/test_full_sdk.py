# test_full_sdk.py — Phase 4 Full SDK Extraction Verification
# 20-point test suite covering all 9 new modules + existing module upgrades

import asyncio
import sys
import os
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.dirname(__file__))

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


async def run_tests():
    global PASS, FAIL

    print("=" * 60)
    print("PHASE 4: Full SDK Extraction - 20-Point Verification")
    print("=" * 60)

    # ── Test 1: llm_types imports and types work ──────────────────
    print("\n[1] llm_types.py")
    try:
        from llm_types import (TextPart, ImageBytesPart, AudioPart, JsonPart,
                               Role, Message, ToolSchema, NormalizedResponse,
                               NormalizedStatus, NormalizedUsage, NormalizedToolCallItem)
        tp = TextPart("hello")
        check("TextPart creation", tp.text == "hello")
        msg = Message(role=Role.USER, content=[tp])
        check("Message with Role", msg.text_content() == "hello")
        ts = ToolSchema(name="test", description="desc", parameters_schema={"type": "object"})
        check("ToolSchema creation", ts.to_dict()["name"] == "test")
        nr = NormalizedResponse(id="r1", model="gpt-4", status=NormalizedStatus.COMPLETED)
        check("NormalizedResponse", nr.status == NormalizedStatus.COMPLETED)
    except Exception as e:
        check(f"llm_types import: {e}", False)

    # ── Test 2: instructions.py ───────────────────────────────────
    print("\n[2] instructions.py")
    try:
        from instructions import Instructions, InstructionPresets
        inst = Instructions("You are a bot.")
        check("Instructions basic parse", len(inst.full_reference) > 0)
        check("InstructionPresets.security_camera", "security" in InstructionPresets.security_camera().lower())
        check("InstructionPresets.fitness_coach", "fitness" in InstructionPresets.fitness_coach().lower() or "coach" in InstructionPresets.fitness_coach().lower())
        custom = InstructionPresets.custom("a helper", ["see", "hear"])
        check("InstructionPresets.custom", "helper" in custom)
    except Exception as e:
        check(f"instructions import: {e}", False)

    # ── Test 3: warmup_cache.py ──────────────────────────────────
    print("\n[3] warmup_cache.py")
    try:
        from warmup_cache import WarmupCache, Warmable, warmup_cache

        class DummyWarmable(Warmable):
            def __init__(self):
                self.resource = None
            async def on_warmup(self):
                return {"loaded": True}
            def on_warmed_up(self, resource):
                self.resource = resource

        cache = WarmupCache()
        dw = DummyWarmable()
        await cache.warmup(dw)
        check("Warmable loads resource", dw.resource == {"loaded": True})
        check("Cache reports stats", cache.get_stats()["total_cached"] == 1)
        # Second warmup should use cache
        dw2 = DummyWarmable()
        await cache.warmup(dw2)
        check("Cache reuses resource", dw2.resource == {"loaded": True})
    except Exception as e:
        check(f"warmup_cache: {e}", False)

    # ── Test 4: profiling.py ─────────────────────────────────────
    print("\n[4] profiling.py")
    try:
        from profiling import Profiler, profiler
        p = Profiler()

        @p.track("test_op")
        async def dummy_op():
            await asyncio.sleep(0.01)
            return 42

        result = await dummy_op()
        check("Profiled async function runs", result == 42)
        stats = p.get_stats("test_op")
        check("Profiler captures timing", stats["count"] == 1 and stats["min_ms"] > 0)

        # Manual timing
        tid = p.start_timer("manual")
        time.sleep(0.005)
        dur = p.stop_timer(tid)
        check("Manual timer works", dur > 0)
    except Exception as e:
        check(f"profiling: {e}", False)

    # ── Test 5: rag_engine.py ────────────────────────────────────
    print("\n[5] rag_engine.py")
    try:
        from rag_engine import SimpleRAG, Document
        rag = SimpleRAG()
        docs = [
            Document(text="YOLO is a real-time object detection system.", source="yolo.md"),
            Document(text="Pose estimation tracks body joints for fitness.", source="pose.md"),
            Document(text="Security cameras monitor for threats.", source="security.md"),
        ]
        n = await rag.add_documents(docs)
        check("RAG indexes documents", n > 0)
        results = await rag.search("object detection")
        check("RAG search finds relevant results", "yolo" in results.lower() or "detection" in results.lower())
        stats = rag.get_stats()
        check("RAG stats correct", stats["total_chunks"] > 0 and len(stats["sources"]) == 3)
    except Exception as e:
        check(f"rag_engine: {e}", False)

    # ── Test 6: transcript_buffer.py ─────────────────────────────
    print("\n[6] transcript_buffer.py")
    try:
        from transcript_buffer import TranscriptBuffer
        buf = TranscriptBuffer()
        buf.push_interim("Hello how are")
        check("Interim push works", "(Hello how are)" in buf.get_current_text())
        buf.push_final("Hello, how are you?")
        check("Final push works", "Hello, how are you?" in buf.get_current_text())
        stats = buf.get_stats()
        check("Buffer stats correct", stats["final_segments"] == 1 and not stats["has_interim"])
    except Exception as e:
        check(f"transcript_buffer: {e}", False)

    # ── Test 7: turn_detection.py ────────────────────────────────
    print("\n[7] turn_detection.py")
    try:
        from turn_detection import SilenceTurnDetector
        td = SilenceTurnDetector(silence_threshold_ms=50, min_speech_duration_ms=10)
        # Simulate speech
        e1 = td.on_audio_level(0.5)
        check("Speech start detected", e1 is not None and e1.type == "start_speaking")
        check("is_speaking = True", td.is_speaking)
        # Feed more speech then silence
        for _ in range(5):
            td.on_audio_level(0.3)
        time.sleep(0.06)
        e2 = td.on_audio_level(0.01)
        stats = td.get_stats()
        check("Turn detector has events", stats["total_events"] >= 1)
    except Exception as e:
        check(f"turn_detection: {e}", False)

    # ── Test 8: mcp_tools.py ─────────────────────────────────────
    print("\n[8] mcp_tools.py")
    try:
        from mcp_tools import MCPManager
        mgr = MCPManager()
        server = await mgr.add_server("test_server", url="http://localhost:9999")
        check("MCP server added", server.is_connected)
        server.register_tool("greet", description="Say hello", parameters_schema={"type": "object"})
        tools = await mgr.list_tools()
        check("MCP tool registered", len(tools) == 1 and tools[0].name == "greet")
        # Register local handler
        mgr.register_tool_handler("local_tool", lambda: {"hello": "world"})
        result = await mgr.call_tool("local_tool", {})
        check("MCP local handler works", result["hello"] == "world")
        stats = mgr.get_stats()
        check("MCP stats correct", stats["total_servers"] == 1)
    except Exception as e:
        check(f"mcp_tools: {e}", False)

    # ── Test 9: agent_core.py ────────────────────────────────────
    print("\n[9] agent_core.py")
    try:
        from agent_core import AgentCore, AgentConfig
        agent = AgentCore(config=AgentConfig(
            name="TestAgent",
            instructions="You are a test agent.",
            enable_tools=False,
        ))
        check("Agent created", agent.config.name == "TestAgent")
        check("Agent has stats", agent.get_stats()["name"] == "TestAgent")

        # Register a test tool
        async def greet(name: str = "World"):
            return f"Hello, {name}!"
        agent.register_tool("greet", greet)
        check("Agent tool registered", "greet" in agent.get_stats()["registered_tools"])
    except Exception as e:
        check(f"agent_core: {e}", False)

    # ── Test 10: function_registry MCP + Enum upgrades ───────────
    print("\n[10] function_registry upgrades")
    try:
        from function_registry import FunctionRegistry
        from enum import Enum

        reg = FunctionRegistry()

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        @reg.register(description="Pick a color")
        async def pick_color(color: Color) -> dict:
            return {"color": color.value}

        schema = reg.get_tools_schema()
        check("Enum in schema", "enum" in str(schema))

        # MCP tool registration
        async def mcp_handler(query: str):
            return {"result": query}
        reg.register_mcp_tool("search_web", "Search the web",
                              {"type": "object", "properties": {"query": {"type": "string"}}},
                              mcp_handler)
        check("MCP tool registered", reg.has("search_web"))
        check("get_callable works", reg.get_callable("search_web") is not None)
        check("list_tools works", "search_web" in reg.list_tools())
    except Exception as e:
        check(f"function_registry upgrades: {e}", False)

    # ── Test 11: existing modules still work ─────────────────────
    print("\n[11] Existing modules (Phase 1-3)")
    try:
        from event_bus import EventBus, EventType, Event
        from observability import ProviderHealthTracker, PlatformMetrics
        from video_processor import create_default_pipeline
        from conversation import ConversationManager

        eb = EventBus()
        received = []
        eb.subscribe(EventType.INFO, lambda e: received.append(e))
        await eb.emit(Event(type=EventType.INFO, data={"msg": "test"}))
        check("EventBus still works", len(received) == 1)

        ht = ProviderHealthTracker()
        ht.record_success("test_prov", 100.0)
        check("HealthTracker still works", ht.get_all_stats()["providers"]["test_prov"]["success_count"] == 1)

        pipeline = create_default_pipeline()
        check("Pipeline still works", pipeline.get_stats()["total_frames"] >= 0)

        cm = ConversationManager()
        s = cm.get_or_create("test_session")
        s.add_message("user", "hello")
        check("ConversationManager still works", s.message_count == 1)
    except Exception as e:
        check(f"existing modules: {e}", False)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"RESULTS: {PASS}/{total} passed, {FAIL} failed")
    if FAIL == 0:
        print("ALL TESTS PASSED - Full SDK extraction verified!")
    else:
        print(f"WARNING: {FAIL} test(s) failed")
    print("=" * 60)
    return FAIL


if __name__ == "__main__":
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)
