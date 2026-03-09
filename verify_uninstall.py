import os
import shutil
from app import clear_cache

# Create a dummy cache file to verify deletion
cache_dir = os.path.expanduser("~/.cache/huggingface")
os.makedirs(cache_dir, exist_ok=True)
dummy_file = os.path.join(cache_dir, "dummy.txt")
with open(dummy_file, "w") as f:
    f.write("test")

print(f"Dummy cache exists: {os.path.exists(dummy_file)}")

result = clear_cache()
print(result)

print(f"Dummy cache still exists: {os.path.exists(dummy_file)}")
if not os.path.exists(dummy_file):
    print("✅ Uninstall/Clear Cache functionality verified.")
else:
    print("❌ Uninstall/Clear Cache functionality FAILED.")
