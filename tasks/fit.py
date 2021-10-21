import pandas as pd

# %% tags=["parameters"]
upstream = ['clean']
product = None

# %%
data = pd.read_csv(upstream['clean']['data'])

# %% id="NQCiGUCudfOA" outputId="63d07d19-d863-4907-d48e-34e61b06ae7b"
# taking the labels out from the data

y = data['target']

data = data.drop('target', axis=1)

print("Shape of y:", y.shape)

# %% id="aol5uGnYM3J0"
# one hot encoding of the data
# drop_first = True, means dropping the first categories from each of the attribues
# for ex gender having gender_male and gender-female would be male having values 1 and 0

data = pd.get_dummies(data, drop_first=True)

# %% id="Z4zceZNLOF-F" outputId="203132c7-55da-416a-933f-ae6d835a65ae"
# checking the dataset after encoding

data.head()

# %% id="D5Ugtg5AOI7b" outputId="0726349e-e509-4f10-85ba-1fff2b466e6d"
# splitting the dependent and independent variables from the data

x = data

# checking the shapes of x and y
print("Shape of x:", x.shape)
print("Shape of y:", y.shape)

# %% id="vUOaL0E0fRFG" outputId="619d9aac-0028-4987-fb7e-3b6820f17ca6"
y.value_counts()

# %% id="iAbN_FvGOq19" outputId="60b2767a-e27d-48c0-bd77-d9279060f79a"
# splitting the sets into training and test sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

# getting the shapes
print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_test :", y_test.shape)

# %% [markdown] id="xKOFulQVQv64"
# Diagnostic tests are often sold, marketed, cited and used with sensitivity and specificity as the headline metrics. Sensitivity and specificity are defined as,
#
# Sensitivity = TruePositives/TruePositives+FalseNegatives
#
# Specificity = FalseNegatives/FalseNegatives+TruePositives

# %% [markdown]
# **Modelling**

# %% [markdown]
# ## Random Forest Classifier

# %% id="blJoElHXPSSF" outputId="460283eb-4334-4acd-dd32-0d6200088d93"
# MODELLING
# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
y_pred_quant = model.predict_proba(x_test)[:, 1]
y_pred = model.predict(x_test)

# %% store outputs
import pickle
from pathlib import Path

Path(product['model']).write_bytes(pickle.dumps(model))
Path(product['x_train']).write_bytes(pickle.dumps(x_train))
Path(product['y_train']).write_bytes(pickle.dumps(y_train))
Path(product['x_test']).write_bytes(pickle.dumps(x_test))
Path(product['y_test']).write_bytes(pickle.dumps(y_test))
Path(product['y_pred']).write_bytes(pickle.dumps(y_pred))
Path(product['y_pred_quant']).write_bytes(pickle.dumps(y_pred_quant))
Path(product['data']).write_bytes(pickle.dumps(data))