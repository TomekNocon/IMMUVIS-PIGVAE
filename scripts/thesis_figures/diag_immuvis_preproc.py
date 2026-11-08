"""Which preprocessing / checkpoint reproduces the stored ImmuVis 16x16x768 embeddings? (szary, GPU)

For a few crops of slide 7064, encode under several variants and correlate with the stored h5 map.
"""
import glob, os, sys
import h5py, numpy as np, tifffile, torch
from skimage.filters import butterworth
sys.path.insert(0, "."); sys.path.insert(0, "scripts/thesis_figures")
import immuvis_encoder as ie

D = "/raid_encrypted/immucan/embeddings/tnocon/data/IMC/cords"
SPLIT = "/raid_encrypted/immucan/immuvis_split"
EMB = "/raid_encrypted/immucan/embeddings/tnocon/ImmuVis-768-UN-475"
IMG = "7064.tiff"; P = 128
dev = "cuda" if torch.cuda.is_available() else "cpu"

print("== files next to embeddings (non-npy):")
for p in sorted(glob.glob(f"{EMB}/**/*", recursive=True)):
    if not p.endswith((".npy", ".csv")): print("  ", p)
print("== model files:")
for p in sorted(glob.glob("/raid_encrypted/immucan/models/*")): print("  ", p)
print("== onlyarcsinh patches dir:")
for p in sorted(glob.glob("/raid_encrypted/immucan/immuvis_split_patches_onlyarcsinh/*"))[:10]: print("  ", p)
for p in sorted(glob.glob("/raid_encrypted/immucan/immuvis_split_patches_onlyarcsinh/*/cords*"))[:5]: print("  ", p)
csvs = sorted(glob.glob(f"{EMB}/*/*metadata_batch_*.csv"))
print("== metadata csv example:", csvs[:1])
if csvs:
    import pandas as pd
    m = pd.read_csv(csvs[0]); print(m.head(3).to_string()); print(m.columns.tolist())

h5 = None
for s in ["train", "test"]:
    f = h5py.File(f"{D}/{s}.h5", "r")
    paths = np.array([p.decode().split("/")[-1] for p in f["paths"][:]])
    rows = np.where(paths == IMG)[0]
    if len(rows):
        h5, split, rows = f, s, rows; print("== stored in", s, "rows", len(rows), "full path", f["paths"][rows[0]].decode()); break
pos = h5["positions"][rows[0]:rows[-1] + 1]
pick = [rows[j] for j in np.linspace(0, len(rows) - 1, 4).astype(int)]

tiff = [q for q in (f"{SPLIT}/{sp}/cords/imgs/{IMG}" for sp in ["train", "test"]) if os.path.exists(q)][0]
raw = tifffile.imread(tiff).astype(np.float32); print("== raw", tiff, raw.shape, raw.dtype, float(raw.max()))
imm, _ = ie.load_immuvis(device=dev)
cids, _ = ie.dataset_channel_ids("cords"); cids = cids.to(dev)

ash = np.arcsinh(raw / 5.0)
bw_whole = np.stack([butterworth(c, cutoff_frequency_ratio=0.2, high_pass=False) for c in ash])
def bw(x): return np.stack([butterworth(c, cutoff_frequency_ratio=0.2, high_pass=False) for c in x])

variants = {
    "A arcsinh>bw(whole)>clip4/4 [current]": lambda a, b: np.clip(bw_whole[:, a:a+P, b:b+P], 0, 4) / 4,
    "B arcsinh>bw(crop)>clip4/4": lambda a, b: np.clip(bw(ash[:, a:a+P, b:b+P]), 0, 4) / 4,
    "C arcsinh only": lambda a, b: ash[:, a:a+P, b:b+P],
    "D arcsinh>clip4/4 (no bw)": lambda a, b: np.clip(ash[:, a:a+P, b:b+P], 0, 4) / 4,
    "E arcsinh>bw(whole) (no clip)": lambda a, b: bw_whole[:, a:a+P, b:b+P],
    "F arcsinh>bw(crop) (no clip)": lambda a, b: bw(ash[:, a:a+P, b:b+P]),
    "G raw (no arcsinh)": lambda a, b: raw[:, a:a+P, b:b+P],
}
res = {k: [] for k in variants}
for r in pick:
    a, b = int(h5["positions"][r][0]), int(h5["positions"][r][1])
    st = torch.from_numpy(h5["embeddings"][r]).float().to(dev)
    for k, fn in variants.items():
        x = torch.from_numpy(np.ascontiguousarray(fn(a, b), dtype=np.float32))[None].to(dev)
        with torch.no_grad():
            lat = ie.encode(imm, x, cids[None])[0].float()
        c = torch.corrcoef(torch.stack([lat.flatten(), st.flatten()]))[0, 1].item()
        res[k].append(c)
    print(f"crop at ({a},{b}) stored mean {st.mean():.3f} std {st.std():.3f}", flush=True)
print("== mean corr with stored embedding over", len(pick), "crops")
for k, v in res.items(): print(f"  {np.mean(v):.4f}  {k}   {np.round(v, 3)}")
