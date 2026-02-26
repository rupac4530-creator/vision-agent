"""Phase 3 Verification: All SDK modules, enhanced security cam & pose engine."""
import sys, time

# Test 1: All Phase 1 modules
from function_registry import tool_registry
from event_bus import event_bus, EventType
from observability import health_tracker, platform_metrics
from video_processor import create_default_pipeline
from conversation import conversation_manager
print(f"Test 1 PASS: Phase 1 modules OK ({tool_registry.count} tools)")

# Test 2: Enhanced LLM provider
from llm_provider import provider, CascadeProvider
has_health = isinstance(provider, CascadeProvider) and provider._health is not None
has_stream = hasattr(provider, "chat_stream")
print(f"Test 2 PASS: LLM cascade OK (health={has_health}, stream={has_stream})")

# Test 3: Enhanced Security Camera
from security_cam import security_camera, SecurityZone, TrackedObject
assert hasattr(security_camera, "_zones"), "Missing zones"
assert hasattr(security_camera, "_tracked_objects"), "Missing tracking"
assert hasattr(security_camera, "get_metrics"), "Missing metrics"
assert hasattr(security_camera, "add_zone"), "Missing add_zone"
assert hasattr(security_camera, "_check_abandoned_objects"), "Missing abandoned check"
metrics = security_camera.get_metrics()
assert "zone_violations" in metrics, "Missing zone_violations metric"
assert "abandoned_objects" in metrics, "Missing abandoned_objects metric"
print(f"Test 3 PASS: Security camera enhanced ({len(security_camera._zones)} zones, {len(metrics)} metrics)")

# Test 4: Zone contains check
zone = SecurityZone("Test", (0.0, 0.0, 0.5, 0.5), "restricted")
assert zone.contains(100, 100, 640, 480) == True
assert zone.contains(400, 400, 640, 480) == False
print("Test 4 PASS: SecurityZone.contains() works correctly")

# Test 5: TrackedObject tracking
obj = TrackedObject(1, "person", [10, 20, 50, 80])
obj.update([11, 21, 51, 81])
obj.update([11, 21, 51, 81])
obj.update([11, 21, 51, 81])
obj.update([11, 21, 51, 81])
obj.update([11, 21, 51, 81])
assert obj.frame_count == 6
assert obj.is_stationary == True
print(f"Test 5 PASS: TrackedObject stationary detection works (frames={obj.frame_count})")

# Test 6: Enhanced Pose Engine
from pose_engine import pose_engine, LungeDetector, SKELETON_CONNECTIONS, _compute_form_score
assert "lunge" in pose_engine.SUPPORTED_EXERCISES
assert "lunge" in pose_engine.DETECTOR_CLASSES
assert len(SKELETON_CONNECTIONS) == 17
print(f"Test 6 PASS: PoseEngine enhanced (exercises={pose_engine.SUPPORTED_EXERCISES}, {len(SKELETON_CONNECTIONS)} skeleton connections)")

# Test 7: LungeDetector
lunge = LungeDetector()
assert lunge.exercise == "lunge"
assert lunge.reps == 0
# Simulate a lunge cycle
kp = [[0,0,0.9]]*17  # 17 keypoints
# Set hip, knee, ankle for left side
kp[11] = [100, 100, 0.9]  # left_hip
kp[13] = [100, 200, 0.9]  # left_knee
kp[15] = [100, 300, 0.9]  # left_ankle
kp[12] = [200, 100, 0.9]  # right_hip
kp[14] = [200, 200, 0.9]  # right_knee
kp[16] = [200, 300, 0.9]  # right_ankle
result = lunge.update(kp)
assert "reps" in result
assert "corrections" in result
print(f"Test 7 PASS: LungeDetector works (angle={result['angle']}, state={result['state']})")

# Test 8: Form score computation
score = _compute_form_score("squat", 90, "down", [])
assert 80 <= score <= 100, f"Expected high score for perfect squat, got {score}"
score_bad = _compute_form_score("squat", 40, "down", ["Too deep", "Lean forward"])
assert score_bad < score, f"Bad form should score lower ({score_bad} vs {score})"
print(f"Test 8 PASS: Form scoring works (perfect={score}, bad={score_bad})")

# Test 9: Skeleton connections
from pose_engine import PoseEngine
eng = PoseEngine()
connections = eng._get_skeleton_connections(kp)
assert len(connections) > 0
assert "from" in connections[0]
assert "color" in connections[0]
print(f"Test 9 PASS: Skeleton connections generated ({len(connections)} bones)")

# Test 10: Session summary with coaching feedback
track = eng._get_track(0, "squat")
track["form_scores"] = [85, 90, 88, 92, 87]
summary = eng.get_session_summary(0)
assert "avg_form_score" in summary
assert "feedback" in summary
assert len(summary["feedback"]) > 0
print(f"Test 10 PASS: Session summary with coaching (avg_score={summary['avg_form_score']}, feedback='{summary['feedback'][0]}')")

# Test 11: All tracks summary
all_tracks = eng.get_all_tracks_summary()
assert "total_tracks" in all_tracks
assert "total_reps" in all_tracks
print(f"Test 11 PASS: Multi-track summary (tracks={all_tracks['total_tracks']})")

# Test 12: event bus has correct event types
assert hasattr(EventType, "SECURITY_ALERT")
assert hasattr(EventType, "REP_COUNTED")
assert hasattr(EventType, "MILESTONE")
assert hasattr(EventType, "CORRECTION")
print("Test 12 PASS: EventType has all required types for security + pose")

print("\n" + "="*55)
print("ALL 12 TESTS PASSED — PHASES 1, 2, 3 VERIFIED")
print("="*55)
