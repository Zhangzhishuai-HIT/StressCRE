"""Frozen PlantCAD2 representations of the 3,020-bp windows (plan step 4, candidate 2).

Last hidden layer (1,536 channels = forward + reverse-complement halves of the RC-equivariant model),
averaged in 50-bp bins separately over the TSS half (bases 0-1499 -> 30 bins) and the TTS half
(bases 1520-3019 -> 30 bins); the 20-N spacer is dropped. Output per chunk of genes:
<out_dir>/chunk_XXXX.npy  float16 [n_genes, 60, 1536]  and  <out_dir>/chunk_XXXX.genes.txt.
Existing chunks are skipped, so the run can be restarted at any time.

Usage: CUDA_VISIBLE_DEVICES=<g> python embed_plantcad2.py <windows.fa.gz> <model_dir> <out_dir> [batch=8] [mem_frac=0.2] [chunk=1000] [shard=0/1]
  shard i/n: this process takes chunks k with k % n == i (run n processes on n GPUs).
Runs: 2026-11-16 chunks 0-5 on gpu10 GPU 0; the rest on AutoDL (3 x RTX 4090).
Fix 2026-11-16: the last batch of a chunk is now clipped at the chunk end. Before, a batch size that does not
divide the chunk size (AutoDL run: batch 16, chunk 1000) wrote 8 extra rows (the next chunk's first genes) at the
end of chunks 0-51; those files were repaired with fix_embed_chunk_rows.py.
"""
import gzip, os, sys, time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

FA, MD, OUT = sys.argv[1:4]
B = int(sys.argv[4]) if len(sys.argv) > 4 else 8
FRAC = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
CH = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
SI, SN = (int(x) for x in (sys.argv[7] if len(sys.argv) > 7 else "0/1").split("/"))
os.makedirs(OUT, exist_ok=True)

names, seqs = [], []
for line in gzip.open(FA, "rt"):
    if line.startswith(">"):
        names.append(line[1:].strip())
    else:
        seqs.append(line.strip())
assert all(len(s) == 3020 for s in seqs)
print(f"{len(names)} windows; batch {B}; chunk {CH}; shard {SI}/{SN}", flush=True)

assert torch.cuda.is_available()
free, total = torch.cuda.mem_get_info()
print(f"GPU free {free / 2**30:.1f} / {total / 2**30:.1f} GiB", flush=True)
torch.cuda.set_per_process_memory_fraction(FRAC)
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
model = AutoModel.from_pretrained(MD, trust_remote_code=True).cuda().eval()

t0 = time.time(); done = 0
for c0 in range(0, len(names), CH):
    k = c0 // CH
    fo = os.path.join(OUT, f"chunk_{k:04d}.npy")
    if k % SN != SI or os.path.exists(fo):
        continue
    e = min(c0 + CH, len(names))
    out = []
    for s in range(c0, e, B):
        ids = tok(seqs[s:min(s + B, e)], return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(input_ids=ids).last_hidden_state.float()           # [b, 3020, 1536]
        tss = h[:, :1500].reshape(h.shape[0], 30, 50, h.shape[2]).mean(2)
        tts = h[:, 1520:].reshape(h.shape[0], 30, 50, h.shape[2]).mean(2)
        out.append(torch.cat([tss, tts], 1).half().cpu().numpy())
    arr = np.concatenate(out)
    assert len(arr) == e - c0, (k, len(arr), e - c0)
    np.save(fo + ".tmp.npy", arr)
    os.replace(fo + ".tmp.npy", fo)
    with open(os.path.join(OUT, f"chunk_{k:04d}.genes.txt"), "w") as g:
        g.write("\n".join(names[c0:e]) + "\n")
    done += len(arr)
    el = time.time() - t0
    print(f"chunk {k}: {len(arr)} genes; {done / el:.2f} genes/s; elapsed {el / 60:.1f} min", flush=True)
print("all chunks done", flush=True)
