"""Check Kaggle kernel status for birdclef-trainv19."""
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Check if kernel exists and get its status
try:
    result = api.kernels_status("hellodave2035/birdclef-trainv19")
    print("kernels_status:", result)
except Exception as e:
    print(f"kernels_status error: {e}")

# Try getting logs
try:
    result = api.kernels_logs("hellodave2035/birdclef-trainv19")
    if result:
        print("--- LOGS ---")
        for line in result[-200:]:
            print(line)
    else:
        print("kernels_logs: empty")
except Exception as e:
    print(f"kernels_logs error: {e}")

# List recent kernels
try:
    result = api.kernels_list(user="hellodave2035", page_size=5, sort_by="dateCreated")
    for k in result:
        print(f"  ref={k.ref}  title={k.title}")
        for attr in dir(k):
            if not attr.startswith("_"):
                try:
                    val = getattr(k, attr)
                    if not callable(val) and val is not None:
                        print(f"    {attr}={val}")
                except:
                    pass
except Exception as e:
    print(f"kernels_list error: {e}")
