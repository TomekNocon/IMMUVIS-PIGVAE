from pathlib import Path
import numpy as np
import pandas as pd

# train_path = Path("/raid_encrypted/immucan/embeddings/tnocon/") / "ImmuVis-768-UN-475" / "train" / "ImmuVis-768-UN-475_train_image_patches_embeddings_batch_0.npy"
train_path = Path("/raid_encrypted/immucan/embeddings/tnocon/") / "ImmuVis-768-UN-475" / "train" / "ImmuVis-768-UN-475_train_image_patches_metadata_batch_10.csv"

print(train_path)

# data = np.load(train_path)

data = pd.read_csv(train_path)
data = data["panel"].value_counts()
print(data)