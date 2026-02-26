# test_deep_sdk.py — Complete 50-point verification for ALL 17 SDK modules
import asyncio, sys, os, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.dirname(__file__))
PASS = FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  [PASS] {name}")
    else: FAIL += 1; print(f"  [FAIL] {name}")

async def run():
    global PASS, FAIL
    print("=" * 60)
    print("DEEP SDK EXTRACTION — 50-Point Verification")
    print("=" * 60)

    # ── 1-4: llm_types ─────────────────────────────────────────
    print("\n[1-4] llm_types.py")
    from llm_types import TextPart, ImageBytesPart, Role, Message, ToolSchema, NormalizedResponse, NormalizedStatus
    check("TextPart", TextPart("hi").text == "hi")
    check("Message", Message(Role.USER, [TextPart("x")]).text_content() == "x")
    check("ToolSchema", ToolSchema("t","d",{}).to_dict()["name"] == "t")
    check("NormalizedResponse", NormalizedResponse("1","m",NormalizedStatus.COMPLETED).status.value == "completed")

    # ── 5-8: instructions ──────────────────────────────────────
    print("\n[5-8] instructions.py")
    from instructions import Instructions, InstructionPresets
    check("Parse text", len(Instructions("Bot system.").full_reference) > 0)
    check("security preset", "security" in InstructionPresets.security_camera().lower())
    check("fitness preset", "coach" in InstructionPresets.fitness_coach().lower() or "fitness" in InstructionPresets.fitness_coach().lower())
    check("custom preset", "helper" in InstructionPresets.custom("a helper", ["see"]))

    # ── 9-11: warmup_cache ─────────────────────────────────────
    print("\n[9-11] warmup_cache.py")
    from warmup_cache import WarmupCache, Warmable
    class DW(Warmable):
        def __init__(self): self.r = None
        async def on_warmup(self): return 42
        def on_warmed_up(self, r): self.r = r
    c = WarmupCache(); d = DW(); await c.warmup(d)
    check("Warmable loads", d.r == 42)
    check("Stats", c.get_stats()["total_cached"] >= 1)
    d2 = DW(); await c.warmup(d2)
    check("Cache reuse", d2.r == 42)

    # ── 12-14: profiling ───────────────────────────────────────
    print("\n[12-14] profiling.py")
    from profiling import Profiler
    p = Profiler()
    @p.track("op")
    async def f(): await asyncio.sleep(0.005); return 1
    check("Track result", await f() == 1)
    check("Timing captured", p.get_stats("op")["count"] == 1)
    tid = p.start_timer("m"); time.sleep(0.003); check("Manual timer", p.stop_timer(tid) > 0)

    # ── 15-17: rag_engine ──────────────────────────────────────
    print("\n[15-17] rag_engine.py")
    from rag_engine import SimpleRAG, Document
    r = SimpleRAG()
    await r.add_documents([Document("YOLO detects objects","a"), Document("Pose tracks joints","b")])
    check("Index", r.get_stats()["total_chunks"] > 0)
    search_result = await r.search("detect objects")
    check("Search", len(search_result) > 0)
    check("Sources", len(r.get_stats()["sources"]) == 2)

    # ── 18-20: transcript_buffer ───────────────────────────────
    print("\n[18-20] transcript_buffer.py")
    from transcript_buffer import TranscriptBuffer
    b = TranscriptBuffer(); b.push_interim("partial")
    check("Interim", "(partial)" in b.get_current_text())
    b.push_final("Final sentence."); check("Final", "Final sentence." in b.get_current_text())
    check("Stats", b.get_stats()["final_segments"] == 1)

    # ── 21-23: turn_detection ──────────────────────────────────
    print("\n[21-23] turn_detection.py")
    from turn_detection import SilenceTurnDetector
    td = SilenceTurnDetector(silence_threshold_ms=50, min_speech_duration_ms=10)
    e = td.on_audio_level(0.5); check("Start speech", e is not None)
    check("Speaking", td.is_speaking)
    for _ in range(3): td.on_audio_level(0.4)
    check("Stats", td.get_stats()["total_events"] >= 1)

    # ── 24-27: mcp_tools ───────────────────────────────────────
    print("\n[24-27] mcp_tools.py")
    from mcp_tools import MCPManager
    m = MCPManager()
    s = await m.add_server("s1", url="http://localhost:9999")
    check("Server add", s.is_connected)
    s.register_tool("t1", description="test", parameters_schema={})
    check("Tool list", len(await m.list_tools()) == 1)
    m.register_tool_handler("h1", lambda: {"ok": True})
    check("Local handler", (await m.call_tool("h1", {}))["ok"])
    check("Stats", m.get_stats()["total_servers"] == 1)

    # ── 28-30: agent_core ──────────────────────────────────────
    print("\n[28-30] agent_core.py")
    from agent_core import AgentCore, AgentConfig
    a = AgentCore(config=AgentConfig(name="T", instructions="x", enable_tools=False))
    check("Agent name", a.config.name == "T")
    async def greet(n="W"): return f"Hi {n}"
    a.register_tool("greet", greet); check("Tool registered", "greet" in a.get_stats()["registered_tools"])
    check("Stats", a.get_stats()["name"] == "T")

    # ── 31-34: function_registry upgrades ──────────────────────
    print("\n[31-34] function_registry.py upgrades")
    from function_registry import FunctionRegistry
    from enum import Enum
    reg = FunctionRegistry()
    class C(Enum): R="red"; B="blue"
    @reg.register(description="color")
    async def pick(c: C): return {"c": c.value}
    check("Enum schema", "enum" in str(reg.get_tools_schema()))
    async def mh(q: str): return {"r": q}
    reg.register_mcp_tool("sw", "search", {"type":"object"}, mh)
    check("MCP tool", reg.has("sw"))
    check("Callable", reg.get_callable("sw") is not None)
    check("list_tools", "sw" in reg.list_tools())

    # ── 35-37: llm_base ────────────────────────────────────────
    print("\n[35-37] llm_base.py")
    from llm_base import LLMBase, LLMResponseEvent, ToolCallEvent
    class DummyLLM(LLMBase):
        async def generate(self, messages, tools=None, **kw):
            return {"text": "hello", "tool_calls": []}
        def get_provider_tool_format(self, schemas): return schemas
    dl = DummyLLM("test")
    check("LLM created", dl.provider_name == "test")
    dl.set_instructions("You are a bot.")
    check("Instructions set", dl.instructions == "You are a bot.")
    dl.register_tool("t", lambda: "ok")
    check("Tool registered", "t" in dl.get_available_tools())

    # ── 38-40: stt_base ────────────────────────────────────────
    print("\n[38-40] stt_base.py")
    from stt_base import STTBase, TranscriptResponse, STTEvent
    class DummySTT(STTBase):
        async def process_audio(self, data, pid=None):
            self._emit_transcript("hello", pid)
            return TranscriptResponse(text="hello")
    ds = DummySTT("test_stt")
    evts = []; ds.on_event(lambda e: evts.append(e))
    await ds.start()
    r = await ds.process_audio(b"\x00" * 320)
    check("STT transcribes", r.text == "hello")
    check("Event emitted", len(evts) == 1)
    check("Stats", ds.get_stats()["total_transcripts"] == 1)

    # ── 41-43: tts_base ────────────────────────────────────────
    print("\n[41-43] tts_base.py")
    from tts_base import TTSBase, TTSAudioChunk
    class DummyTTS(TTSBase):
        async def synthesize(self, text, **kw):
            yield b"\x00" * 3200  # 100ms of 16kHz mono s16
    dt = DummyTTS("test_tts")
    chunks = await dt.speak("hello")
    check("TTS synthesizes", len(chunks) >= 1)
    check("Chunk data", len(chunks[0].data) == 3200)
    check("Stats", dt.get_stats()["total_syntheses"] == 1)

    # ── 44-46: vad ─────────────────────────────────────────────
    print("\n[44-46] vad.py")
    from vad import EnergyVAD
    v = EnergyVAD(threshold=0.01)
    # Loud audio (speech)
    loud = (16000).to_bytes(2, 'little', signed=True) * 160
    r1 = v.process(loud)
    check("VAD detects speech", r1.is_speech)
    # Quiet audio (silence)
    quiet = b"\x00\x00" * 160
    for _ in range(10): v.process(quiet)
    check("VAD detects silence", not v.is_speech)
    check("Stats", v.get_stats()["frame_count"] >= 2)

    # ── 47-49: edge_types + http_transport + config ────────────
    print("\n[47-49] edge_types, http_transport, config")
    from edge_types import Participant, PcmData, VideoFrame, AudioFormat
    p = Participant(id="u1", name="User", role="user")
    check("Participant", p.to_dict()["name"] == "User")

    from http_transport import HTTPTransport
    ht = HTTPTransport()
    ht.add_processor(lambda data, ct, meta: {"detections": [{"label": "test"}]})
    cr = await ht.process_chunk(b"fake_video")
    check("Transport processes", len(cr.detections) == 1)

    from config import Config
    check("Config loads", Config.to_dict()["features"]["rag"] is not None)

    # ── 50: event_manager ──────────────────────────────────────
    print("\n[50] event_manager.py")
    from event_manager import EventManager, BaseEvent, ToolStartEvent
    em = EventManager()
    em.register(ToolStartEvent)
    received = []; em.on(ToolStartEvent, lambda e: received.append(e))
    em.send(ToolStartEvent(tool_name="test"))
    check("EventManager dispatch", len(received) == 1 and received[0].tool_name == "test")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS}/{PASS+FAIL} passed, {FAIL} failed")
    if FAIL == 0: print("ALL 50 TESTS PASSED — Complete SDK extraction verified!")
    else: print(f"WARNING: {FAIL} test(s) failed")
    print("=" * 60)
    return FAIL

if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
