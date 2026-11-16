"""Download external files for StressCRE (direct connection, never overwrite, MANIFEST.tsv per folder).

Input: a TSV with columns  url<TAB>dest_dir<TAB>file_name[<TAB>expected_sha256]  (no header).
Each file goes to <dest_dir>/<file_name>; an existing file is skipped (its sha256 is still checked if
expected). One MANIFEST.tsv row per new file: file, source_url, downloaded, sha256, note.

Usage: python3 -I fetch_external.py <list.tsv> [n_threads]
"""
import concurrent.futures as cf
import datetime, hashlib, os, sys, threading, urllib.request

LIST = sys.argv[1]
NT = int(sys.argv[2]) if len(sys.argv) > 2 else 4
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never the VPS proxy
lock = threading.Lock()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest_row(d, row):
    m = os.path.join(d, "MANIFEST.tsv")
    with lock:
        new = not os.path.exists(m)
        with open(m, "a") as o:
            if new:
                o.write("file\tsource_url\tdownloaded\tsha256\tnote\n")
            o.write("\t".join(row) + "\n")


def fetch(item):
    url, d, name = item[:3]
    exp = item[3] if len(item) > 3 else ""
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, name)
    if os.path.exists(out):
        h = sha256(out)
        return f"skip (exists) {name}" + ("" if not exp or h == exp else "  SHA256 MISMATCH")
    tmp = out + ".part"
    for attempt in range(4):
        try:
            with opener.open(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as r, open(tmp, "wb") as o:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    o.write(b)
            break
        except Exception as e:
            err = e
    else:
        return f"FAILED {name}: {err}"
    h = sha256(tmp)
    if exp and h != exp:
        os.remove(tmp)
        return f"FAILED {name}: sha256 {h} != expected {exp}"
    os.replace(tmp, out)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    manifest_row(d, [name, url, f"{ts} (cluster login node, direct, no proxy)", h,
                     "sha256 matches the source checksum" if exp else ""])
    return f"ok {name} {os.path.getsize(out)} bytes"


items = [l.rstrip("\n").split("\t") for l in open(LIST) if l.strip()]
with cf.ThreadPoolExecutor(NT) as ex:
    res = list(ex.map(fetch, items))
fails = [r for r in res if r.startswith("FAILED") or "MISMATCH" in r]
print(f"{len(res)} files: {sum(r.startswith('ok') for r in res)} downloaded, "
      f"{sum(r.startswith('skip') for r in res)} already present, {len(fails)} failed")
for r in fails:
    print(r)
