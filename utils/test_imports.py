import sys
print("Testing imports...", flush=True)
for mod in ['pandas', 'numpy', 'scipy', 'sklearn', 'matplotlib']:
    try:
        __import__(mod)
        print(f"  {mod}: OK", flush=True)
    except Exception as e:
        print(f"  {mod}: FAILED - {e}", flush=True)
