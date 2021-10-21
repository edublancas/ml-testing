from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# %% tags=["parameters"]
upstream = ['fit']
product = None


# %%
model = pickle.loads(Path(upstream['fit']['model']).read_bytes())
x_train = pickle.loads(Path(upstream['fit']['x_train']).read_bytes())
y_train = pickle.loads(Path(upstream['fit']['y_train']).read_bytes())
x_test = pickle.loads(Path(upstream['fit']['x_test']).read_bytes())
y_test = pickle.loads(Path(upstream['fit']['y_test']).read_bytes())
data = pickle.loads(Path(upstream['fit']['data']).read_bytes())
y_pred = pickle.loads(Path(upstream['fit']['y_pred']).read_bytes())
y_pred_quant = pickle.loads(Path(upstream['fit']['y_pred_quant']).read_bytes())

# %% evaluating the model
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))

# %% confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot=True, annot_kws={'size': 15}, cmap='PuBu')

# %% classification report
cr = classification_report(y_test, y_pred)
print(cr)

# %% [markdown]
# **Tree for Model Explanation**

# %% _kg_hide-input=true id="6wNEVTGwSFSS" outputId="e2675e9a-04bc-4984-9865-6658e7892511"
from sklearn.tree import export_graphviz

estimator = model.estimators_[1]
feature_names = [i for i in x_train.columns]

y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values

export_graphviz(estimator,
                out_file='tree.dot',
                feature_names=feature_names,
                class_names=y_train_str,
                rounded=True,
                proportion=True,
                label='root',
                precision=2,
                filled=True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=50'])

from IPython.display import Image
Image(filename='tree.png')

# %% [markdown] id="M7hZyrUnVIDw"
# **Specificity and Sensitivity**

# %% id="9hNFsTsPShN6" outputId="91c55840-76a3-48d1-e39a-4401fa62d47a"
total = sum(sum(cm))

sensitivity = cm[0, 0] / (cm[0, 0] + cm[1, 0])
print('Sensitivity : ', sensitivity)

specificity = cm[1, 1] / (cm[1, 1] + cm[0, 1])
print('Specificity : ', specificity)

# %% id="z7tg0EJlWAyJ" outputId="10a7e16e-6d3c-43b8-cba0-2943a1582fb2"
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="-", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.rcParams['figure.figsize'] = (15, 5)
plt.title('ROC curve for diabetes classifier', fontweight=30)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()

# %% id="yvh0qsm-WYum" outputId="d767c29d-75e8-487e-9730-b390a6547a83"
# let's check the auc score

from sklearn.metrics import auc
auc = auc(fpr, tpr)
print("AUC Score :", auc)

# %% [markdown] id="1si59g9fXcQE"
# ## Model Explanation

# %% id="q9HTh4AzXSGN"
# importing ML Explanability Libraries

#for purmutation importance
import eli5
from eli5.sklearn import PermutationImportance

#for SHAP values
import shap
from pdpbox import pdp, info_plots  #for partial plots

# %% [markdown]
# **Eli5 Values**

# %% _kg_hide-input=true id="arM0hk6CYNRy" outputId="c9c0cf83-694e-42d9-c63e-a6a255f3b66a"
# let's check the importance of each attributes

perm = PermutationImportance(model, random_state=0).fit(x_test, y_test)
eli5.show_weights(perm, feature_names=x_test.columns.tolist())

# %% [markdown] id="7GizL0WEcFbr"
# **Partial Dependence Plot for Top 5 Features**

# %% _kg_hide-input=true id="Vwoe1xOKbFBL" outputId="e0615292-a163-4adf-f0d9-96e81bc7ed87"
# plotting the partial dependence plot for num_major_vessels

base_features = data.columns.values.tolist()

feat_name = 'num_major_vessels'
pdp_dist = pdp.pdp_isolate(model=model,
                           dataset=x_test,
                           model_features=base_features,
                           feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

# %% _kg_hide-input=true id="DVOfQUXrcRzO" outputId="2db8f7d6-46d9-4330-f819-8f96cd8547a1"
# let's plot the partial dependence plot for thalassemia_fixed defect

base_features = data.columns.values.tolist()

feat_name = 'thalassemia_fixed defect'
pdp_dist = pdp.pdp_isolate(model=model,
                           dataset=x_test,
                           model_features=base_features,
                           feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

# %% _kg_hide-input=true id="TnEBkXd0hN2x" outputId="882b6c6d-d926-4a29-c3b8-1d449dfacbcf"

# let's plot the partial dependence plot for thalassemia_reversable defect

base_features = data.columns.values.tolist()

feat_name = 'thalassemia_reversable defect'
pdp_dist = pdp.pdp_isolate(model=model,
                           dataset=x_test,
                           model_features=base_features,
                           feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

# %% _kg_hide-input=true id="iw5qC8FbhcxN" outputId="51397baa-d1e9-40b2-8c77-ee8ccd952733"
# plotting a partial dependence graph for chest_pain_type_atypical angina

base_features = data.columns.values.tolist()

feat_name = 'chest_pain_type_atypical angina'
pdp_dist = pdp.pdp_isolate(model=model,
                           dataset=x_test,
                           model_features=base_features,
                           feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

# %% _kg_hide-input=true id="qvrDYACPh0qr" outputId="04bdbef1-7e79-4b46-ae72-7e6e81587c7b"
# plotting a partial dependence graph for st_depression

base_features = data.columns.values.tolist()

feat_name = 'st_depression'
pdp_dist = pdp.pdp_isolate(model=model,
                           dataset=x_test,
                           model_features=base_features,
                           feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

# %% [markdown]
# **Shap Values**

# %% _kg_hide-input=true id="puWnCatMj5K0" outputId="5c65d033-c159-44da-d206-fb0c072d6e0e"
# let's see the shap values

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test)

shap.summary_plot(shap_values[1], x_test, plot_type="bar")

# %% [markdown]
# **Shap Values for Model Explanation**

# %% _kg_hide-input=true id="FU5hJL-Epz12" outputId="16e51782-f62a-4de4-c4da-c5c8cee63fc0"
shap.summary_plot(shap_values[1], x_test)

# %% _kg_hide-input=true id="C_uUN8Wdp89g"
# let's create a function to check the patient's conditions


def patient_analysis(model, patient):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1],
                           patient)


# %% [markdown] id="hUL8TpYPr2iX"
# **Report for the First Patient**

# %% _kg_hide-input=true id="S8fgkfytrS5T" outputId="6bfc1189-fb5a-4945-8f05-4d5dff95509a"
# let's do some real time prediction for patients

patients = x_test.iloc[1, :].astype(float)
patient_analysis(model, patients)

# %% [markdown] id="LbLQnBEfr6ez"
# **Report for the Second Patient**

# %% _kg_hide-input=true id="u0C_dGsdry5O" outputId="d335ad96-0bac-4257-fa15-a8890f968a0e"
patients = x_test.iloc[:, 2].astype(float)
patient_analysis(model, patients)

# %% [markdown] id="xa4ebhIIsKYp"
# **Report for the Third Patient**

# %% _kg_hide-input=true id="UW5BCyV8sGh0" outputId="f7d90392-7de6-4a35-b0d9-c6dcf6cdb52e"
patients = x_test.iloc[:, 3].astype(float)
patient_analysis(model, patients)

# %% id="v-mjo5JnsW-Z" outputId="47d8313c-9535-4740-995c-9999c213b24e"
# dependence plot

shap.dependence_plot('num_major_vessels',
                     shap_values[1],
                     x_test,
                     interaction_index="st_depression")

# %% [markdown] id="Ye8I2f1Rs-dW"
# **Force Plot**

# %% id="sRgtSI4Js5Uc" outputId="a9413b7d-0273-40ee-a708-e10fa14c3eb7"
shap_values = explainer.shap_values(x_train.iloc[:50])
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], x_test.iloc[:50])

# %%
