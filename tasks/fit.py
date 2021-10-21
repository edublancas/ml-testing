import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %% tags=["parameters"]
upstream = None
product = None
path_to_data = None

# %%
data = pd.read_csv(path_to_data)

# %% _kg_hide-output=true id="v9RUg4_n80k5" outputId="0cc3839f-0922-481e-85d3-b33edbf488c9"
# let's change the names of the  columns for better understanding

data.columns = [
    'age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol',
    'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
    'exercise_induced_angina', 'st_depression', 'st_slope',
    'num_major_vessels', 'thalassemia', 'target'
]

data.columns

# %% _kg_hide-output=true id="JkA3_NBf_CnJ" outputId="913a92cd-1989-4347-a30b-67d88a7b5b22"
data['sex'][data['sex'] == 0] = 'female'
data['sex'][data['sex'] == 1] = 'male'

data['chest_pain_type'][data['chest_pain_type'] == 1] = 'typical angina'
data['chest_pain_type'][data['chest_pain_type'] == 2] = 'atypical angina'
data['chest_pain_type'][data['chest_pain_type'] == 3] = 'non-anginal pain'
data['chest_pain_type'][data['chest_pain_type'] == 4] = 'asymptomatic'

data['fasting_blood_sugar'][data['fasting_blood_sugar'] ==
                            0] = 'lower than 120mg/ml'
data['fasting_blood_sugar'][data['fasting_blood_sugar'] ==
                            1] = 'greater than 120mg/ml'

data['rest_ecg'][data['rest_ecg'] == 0] = 'normal'
data['rest_ecg'][data['rest_ecg'] == 1] = 'ST-T wave abnormality'
data['rest_ecg'][data['rest_ecg'] == 2] = 'left ventricular hypertrophy'

data['exercise_induced_angina'][data['exercise_induced_angina'] == 0] = 'no'
data['exercise_induced_angina'][data['exercise_induced_angina'] == 1] = 'yes'

data['st_slope'][data['st_slope'] == 1] = 'upsloping'
data['st_slope'][data['st_slope'] == 2] = 'flat'
data['st_slope'][data['st_slope'] == 3] = 'downsloping'

data['thalassemia'][data['thalassemia'] == 1] = 'normal'
data['thalassemia'][data['thalassemia'] == 2] = 'fixed defect'
data['thalassemia'][data['thalassemia'] == 3] = 'reversable defect'

# %% id="_SEBqHQGbXlh"
data['sex'] = data['sex'].astype('object')
data['chest_pain_type'] = data['chest_pain_type'].astype('object')
data['fasting_blood_sugar'] = data['fasting_blood_sugar'].astype('object')
data['rest_ecg'] = data['rest_ecg'].astype('object')
data['exercise_induced_angina'] = data['exercise_induced_angina'].astype(
    'object')
data['st_slope'] = data['st_slope'].astype('object')
data['thalassemia'] = data['thalassemia'].astype('object')

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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
y_pred_quant = model.predict_proba(x_test)[:, 1]
y_pred = model.predict(x_test)

# evaluating the model
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))

# cofusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot=True, annot_kws={'size': 15}, cmap='PuBu')

# classification report
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
