from pathlib import Path

import pickle
import pandas as pd

# %% tags=["parameters"]
upstream = ['clean']
product = None
path_to_model = None

# %%
data = pd.read_csv(upstream['clean']['data'])
data = pd.get_dummies(data, drop_first=True)
model = pickle.loads(Path(path_to_model).read_bytes())

# %%
preds = model.predict(data)
pd.DataFrame({'pred': preds}).to_csv(product['data'])