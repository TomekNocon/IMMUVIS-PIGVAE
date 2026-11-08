"""Which ImmuVis checkpoint/code reproduces the stored 16x16x768 embeddings? (szary, GPU)

Compares, on 4 crops of 7064, the stored h5 map against:
  old  : submodule code + ImmuVis-768_model-UN-475.pth (current loader)
  refac: colleague's refactored code (~/multiplex-image-model) + ImmuVis-475-768-ma1-refactored.pth
"""
import os, sys
import h5py, numpy as np, tifffile, torch, yaml
sys.path.insert(0, "."); sys.path.insert(0, "scripts/thesis_figures")
import immuvis_encoder as ie
from fig_cords_crop import preprocess

D = "/raid_encrypted/immucan/embeddings/tnocon/data/IMC/cords"
M = "/raid_encrypted/immucan/models"
IMG = "7064.tiff"; P = 128
dev = "cuda" if torch.cuda.is_available() else "cpu"

f = h5py.File(f"{D}/train.h5", "r")
paths = np.array([p.decode().split("/")[-1] for p in f["paths"][:]]); rows = np.where(paths == IMG)[0]
pick = [rows[j] for j in np.linspace(0, len(rows) - 1, 4).astype(int)]
pre = preprocess(tifffile.imread(f"/raid_encrypted/immucan/immuvis_split/train/cords/imgs/{IMG}")).astype(np.float32)
X = torch.from_numpy(np.stack([pre[:, int(f["positions"][r][0]):int(f["positions"][r][0]) + P,
                                    int(f["positions"][r][1]):int(f["positions"][r][1]) + P] for r in pick])).to(dev)
S = torch.from_numpy(np.stack([f["embeddings"][r] for r in pick])).float().to(dev)
print("stored", tuple(S.shape), f"mean {S.mean():.4f} std {S.std():.4f}", flush=True)


def report(name, lat):
    lat = lat.float()
    cs = [torch.corrcoef(torch.stack([lat[k].flatten(), S[k].flatten()]))[0, 1].item() for k in range(len(S))]
    print(f"{name:40s} shape {tuple(lat.shape)} mean {lat.mean():.4f} std {lat.std():.4f} "
          f"corr {np.round(cs, 4)} maxabs {(lat - S).abs().max().item():.4g}", flush=True)


# ---- old loader
imm, (mi, un) = ie.load_immuvis(device=dev)
cids, mn = ie.dataset_channel_ids("cords"); cids = cids.to(dev)
with torch.no_grad():
    report("old UN-475 fp32", ie.encode(imm, X, cids[None].expand(len(X), -1)))
    with torch.autocast("cuda", dtype=torch.float16):
        report("old UN-475 autocast fp16", ie.encode(imm, X, cids[None].expand(len(X), -1)))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        report("old UN-475 autocast bf16", ie.encode(imm, X, cids[None].expand(len(X), -1)))
    for bs in [1]:
        report("old UN-475 batch=1", torch.cat([ie.encode(imm, X[k:k + 1], cids[None]) for k in range(len(X))]))
del imm; torch.cuda.empty_cache()

# ---- refactored
REPO = "/home/tnocon/multiplex-image-model"
sys.path.insert(0, REPO)
try:
    from multiplex_model.modules.immuvis import MultiplexAutoencoder as MA
    ck = torch.load(f"{M}/ImmuVis-475-768-ma1-refactored.pth", map_location="cpu", weights_only=False)
    print("refac ckpt keys:", list(ck.keys()) if isinstance(ck, dict) else type(ck), flush=True)
    cfg = ck.get("model_config")
    if cfg is None:
        y = yaml.safe_load(open(f"{M}/config.ImmuVis-475-768-ma1-refactored.yaml"))
        print("refac yaml top keys:", list(y.keys()), flush=True)
        enc, dec = dict(y["encoder"]), dict(y["decoder"])
        if "hyperkernel" in enc: enc["hyperkernel_config"] = enc.pop("hyperkernel")
        if "hyperkernel" in dec: dec["hyperkernel_config"] = dec.pop("hyperkernel")
        cfg = dict(num_channels=y.get("num_channels", 265), encoder_config=enc, decoder_config=dec)
        for k in ("input_image_size",):
            if k in y: cfg[k] = y[k]
    print("refac model_config:", {k: (v if not isinstance(v, dict) else list(v)) for k, v in cfg.items()}, flush=True)
    model = MA(**cfg)
    r = model.load_state_dict(ck["model_state_dict"], strict=False)
    print("refac missing/unexpected:", len(r.missing_keys), len(r.unexpected_keys), r.missing_keys[:5], r.unexpected_keys[:5], flush=True)
    model.to(dev).eval()
    # tokenizer of the refactored repo
    tok_path = os.path.join(REPO, "configs/all_markers_tokenizer.yaml")
    tok = yaml.safe_load(open(tok_path))
    ids = torch.tensor([tok[m] for m in mn], device=dev)
    print("tokenizer ids equal to old:", bool((ids == cids).all()), flush=True)
    with torch.no_grad():
        report("refac 475 fp32", model.encode(X, ids[None].expand(len(X), -1))["output"])
except Exception as e:
    import traceback; traceback.print_exc(); print("refac FAILED:", repr(e), flush=True)
print("done", flush=True)
