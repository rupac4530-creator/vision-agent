"""Start the Vision Agent server with error logging."""
import os
import sys
import traceback

# Ensure ffmpeg is on PATH
choco_bin = r"C:\ProgramData\chocolatey\bin"
if choco_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = choco_bin + ";" + os.environ.get("PATH", "")

print("=" * 60, flush=True)
print("Vision Agent — Server Startup", flush=True)
print("=" * 60, flush=True)
print(f"Python: {sys.executable}", flush=True)
print(f"CWD:    {os.getcwd()}", flush=True)

# Check ffmpeg
import shutil
ff = shutil.which("ffmpeg")
print(f"ffmpeg: {ff or 'NOT FOUND'}", flush=True)

# Try import
try:
    print("\nImporting main...", flush=True)
    import main
    print("Import OK!", flush=True)
except Exception as e:
    print(f"\nIMPORT ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# Start server
print("\nStarting uvicorn on http://0.0.0.0:8000 ...\n", flush=True)
try:
    import uvicorn
    uvicorn.run(main.app, host="0.0.0.0", port=8000, log_level="info")
except Exception as e:
    print(f"\nSERVER ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
