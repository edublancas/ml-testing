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
from sample import sample

# %% tags=["parameters"]
upstream = None
product = None

# %%
# %%
df = sample(n=5)
# corrupt the data, add a new column
df['unknown_column'] = 1
df.to_csv(product['data'], index=False)