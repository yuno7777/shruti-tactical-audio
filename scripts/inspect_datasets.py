import urllib.request
import json

def inspect_hf_repo(repo_id):
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print(f"--- Files in {repo_id} ---")
            total_bytes = 0
            for item in data:
                path = item.get('path', '')
                size = item.get('size', 0)
                total_bytes += size
                print(f"  {path:50s} : {size / (1024*1024):8.2f} MB")
            print(f"Total: {total_bytes / (1024*1024*1024):.2f} GB")
    except Exception as e:
        print(f"Error inspecting {repo_id}: {e}")

if __name__ == "__main__":
    inspect_hf_repo("Panav-Payappagoudar/sih-26-processed-audio")
