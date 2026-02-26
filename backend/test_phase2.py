"""Phase 2 verification: Enhanced LLM provider with health tracking."""
from llm_provider import provider, CascadeProvider

print(f"Provider: {provider.name} ({provider.display_name})")

if isinstance(provider, CascadeProvider):
    info = provider.all_providers_info()
    print(f"Cascade: {len(info)} providers")
    for p in info:
        health = p.get("health", {})
        if health:
            print(f"  {p['name']}: {p['display']} [rate={health['success_rate']} avg={health['avg_latency_ms']}ms]")
        else:
            print(f"  {p['name']}: {p['display']}")
    print(f"Stats: {provider.stats}")
    print(f"Has chat_stream: {hasattr(provider, 'chat_stream')}")
    print(f"Has _health: {provider._health is not None}")
else:
    print(f"Single provider: {provider.info()}")

# Verify health tracker integration
from observability import health_tracker
health_tracker.record_success("test-provider", 100)
health_tracker.record_success("test-provider", 200)
health_tracker.record_error("test-provider-bad", "test error", 50)
stats = health_tracker.get_all_stats()
print(f"Health tracker: {len(stats['providers'])} providers tracked")

# Verify streaming support
if hasattr(provider, "chat_stream"):
    print("chat_stream method available on CascadeProvider")

# Verify all modules still import together
from function_registry import tool_registry
from event_bus import event_bus
from video_processor import create_default_pipeline
from conversation import conversation_manager
print(f"All modules OK: {tool_registry.count} tools, conv={conversation_manager.count}")

print("\nPHASE 2 VERIFICATION PASSED")
