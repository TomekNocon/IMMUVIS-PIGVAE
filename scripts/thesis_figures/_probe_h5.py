import h5py, glob, numpy as np
d="/raid_encrypted/immucan/embeddings/tnocon/data/IMC/cords"
print(sorted(glob.glob(d+"/*")))
for split in ["train","val","test"]:
    p=f"{d}/{split}.h5"
    try: f=h5py.File(p,"r")
    except Exception as e: print(split,e); continue
    print(split,{k:(f[k].shape,f[k].dtype) for k in f.keys()})
    for k in f.keys():
        if f[k].dtype.kind in "SOU" or f[k].ndim==1: print(k, f[k][:5])
    for k in f.keys():
        if f[k].dtype.kind in "SOU":
            v=np.array([x.decode() if isinstance(x,bytes) else str(x) for x in f[k][:]])
            m=np.where(np.char.find(v,"7775")>=0)[0]; print(k,"7775 rows",m[:40], v[m[:5]])
