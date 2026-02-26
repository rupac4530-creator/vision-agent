"""Quick test: verify /ask endpoint and job polling work."""
import requests
import time
import json

BASE = "http://localhost:8000"

# Test 1: /ask endpoint (2-tier agent)
print("=" * 50)
print("TEST 1: /ask endpoint")
r = requests.post(f"{BASE}/ask", json={"video_stem": "video", "question": "what do you see?"})
data = r.json()
print(f"  Status: {r.status_code}")
print(f"  Has fast_reply: {'fast_reply' in data}")
print(f"  Has job_id: {'job_id' in data}")
print(f"  Fast reply source: {data.get('fast_reply', {}).get('source')}")
print(f"  Fast reply text: {data.get('fast_reply', {}).get('reply', '')[:100]}")
job_id = data.get("job_id")
print(f"  Job ID: {job_id}")

# Test 2: Poll the job
print("\n" + "=" * 50)
print("TEST 2: Job polling")
time.sleep(4)
r2 = requests.get(f"{BASE}/jobs/{job_id}")
j = r2.json()
print(f"  Job status: {j.get('status')}")
print(f"  Job progress: {j.get('progress')}")
print(f"  Job step: {j.get('step')}")
if j.get("result"):
    print(f"  Result source: {j['result'].get('source')}")
    print(f"  Result reply: {j['result'].get('reply', '')[:100]}")
if j.get("error"):
    print(f"  Error: {j.get('error')}")

# Test 3: Homepage loads
print("\n" + "=" * 50)
print("TEST 3: Homepage")
r3 = requests.get(BASE)
print(f"  Status: {r3.status_code}")
print(f"  Has cosmicCanvas: {'cosmicCanvas' in r3.text}")
print(f"  Has XHR progress: {'xhr.upload.onprogress' in r3.text}")
print(f"  Has initCosmic: {'initCosmic' in r3.text}")
print(f"  No bg-glow: {'bg-glow' not in r3.text}")

print("\n" + "=" * 50)
print("ALL TESTS COMPLETE")
