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
import shutil

# %% tags=["parameters"]
upstream = None
product = None
path_to_data = None

# %%
shutil.copy(path_to_data, product['data'])