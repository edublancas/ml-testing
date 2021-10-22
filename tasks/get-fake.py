# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from numpy import random
from sample import sample

# %% tags=["parameters"]
upstream = None
product = None
type_ = None

# %%
# ensure we always generate the same observations, we need this for the
# test_train_serve_skew.py
random.seed(42)
df = sample(n=5)

# corrupt the data
if type_ == 'fake_column':
    df['unknown_column'] = 1
elif type_ == 'fake_training':
    df['target'] = 0
elif type_ is None:
    pass
else:
    raise ValueError('Choose between fake_column, fake_training or None')

df.to_csv(product['data'], index=False)