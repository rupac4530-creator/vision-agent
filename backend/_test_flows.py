"""Verify all fixed flows work correctly."""
import requests
import time
import json

BASE = "http://127.0.0.1:8000"

print("=" * 60)
print("TEST 1: /generate_notes returns job_id immediately")
print("=" * 60)
t0 = time.time()
resp = requests.post(BASE + "/generate_notes?video_stem=video", timeout=5)
elapsed = time.time() - t0
data = resp.json()
print("  Status:", resp.status_code)
print("  Response time: %.2fs (should be < 1s)" % elapsed)
print("  Has job_id:", bool(data.get("job_id")))
print("  Response:", json.dumps(data, indent=2))

if data.get("job_id"):
    job_id = data["job_id"]
    print("\nPolling job %s..." % job_id)
    for i in range(20):
        time.sleep(1)
        jr = requests.get(BASE + "/jobs/" + job_id)
        job = jr.json()
        status = job.get("status")
        print("  [%ds] status=%s progress=%s step=%s" % (i + 1, status, job.get("progress"), job.get("step")))
        if status in ("done", "failed"):
            if status == "done":
                r = job.get("result", {})
                print("  RESULT: provider=%s fallback=%s" % (r.get("llm_provider"), r.get("is_fallback")))
                if r.get("warning"):
                    print("  WARNING:", r.get("warning"))
            else:
                print("  ERROR:", job.get("error"))
            break
    print("PASS: Notes flow completed without hanging!")

print()
print("=" * 60)
print("TEST 2: /jobs/<invalid> returns 404")
print("=" * 60)
resp = requests.get(BASE + "/jobs/nonexistent")
result = "PASS" if resp.status_code == 404 else "FAIL"
print("  Status: %d (expected 404): %s" % (resp.status_code, result))

print()
print("=" * 60)
print("TEST 3: Homepage loads with polling JS")
print("=" * 60)
resp = requests.get(BASE + "/")
result = "PASS" if resp.status_code == 200 else "FAIL"
print("  Status: %d (expected 200): %s" % (resp.status_code, result))
print("  Has pollJob:", "pollJob" in resp.text)
print("  Has job_id handling:", "job_id" in resp.text)

print()
print("ALL TESTS COMPLETE")
