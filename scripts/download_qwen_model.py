"""Download Qwen3-Embedding-0.6B weights from the CN mirror (proxy bypassed)."""
import os
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "_work", "qwen3_embedding_0p6b")
os.makedirs(OUT, exist_ok=True)

BASE = "https://hf-mirror.com/Qwen/Qwen3-Embedding-0.6B/resolve/main/"
MIRROR = "https://alpha-ollama.hf-mirror.com/Qwen/Qwen3-Embedding-0.6B/resolve/main/"

FILES = [
    "config.json", "generation_config.json", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "merges.txt", "special_tokens_map.json", "added_tokens.json",
    "modules.json", "1_Pooling/config.json",
    "config_sentence_transformers.json", "README.md",
]

s = requests.Session()
s.trust_env = False
s.proxies = {"http": "", "https": ""}

for fn in FILES:
    dest = os.path.join(OUT, fn)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"SKIP {fn} ({os.path.getsize(dest)} bytes)")
        continue
    for base in (BASE, MIRROR):
        try:
            r = s.get(base + fn, stream=True, timeout=(20, 180),
                      headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                n = 0
                with open(dest, "wb") as f:
                    for c in r.iter_content(1 << 20):
                        if c:
                            f.write(c)
                            n += len(c)
                r.close()
                print(f"OK   {fn:36s} {n:>12,d} bytes  <- {base.split('/')[2]}")
                break
            else:
                print(f"  {r.status_code} {fn} from {base.split('/')[2]}")
        except Exception as e:
            print(f"  ERR {fn} {type(e).__name__}: {str(e)[:80]}")
    else:
        print(f"MISS {fn}")

print("\n=== files ===")
for dirpath, _, names in os.walk(OUT):
    for nm in names:
        p = os.path.join(dirpath, nm)
        print(f"  {os.path.getsize(p):>12,d}  {os.path.relpath(p, OUT)}")
