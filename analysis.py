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

# %% [markdown]
# **Heart Diseases Analysis**

# %% [markdown]
# <img src="http://www.mankatoclinic.com/stuff/contentmgr/files/0/50f64ba40de262bb3a6bd3e6f50bc9de/image/12_children.jpg" width="700px">

# %% [markdown]
# **Importing Libraries**

# %% _kg_hide-input=true _kg_hide-output=true id="lKaO91WT78vn"
# for basic operations
import numpy as np
import pandas as pd
import pandas_profiling

# for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# for advanced visualizations 
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected = True)
from bubbly.bubbly import bubbleplot

# for model explanation
import shap

# %% tags=["parameters"]
upstream = None
product = None

# %% _kg_hide-input=true _kg_hide-output=true id="0No8Wdtb8Mlw" outputId="b883289b-5690-4abb-bd4d-0cfc3fab5fa8"
# reading the data
data = pd.read_csv('raw.csv')

# getting the shape
data.shape

# %% [markdown] id="9qB486Nl_8Qi"
# **Data Description**

# %% [markdown] id="JPYLs7yt_oRZ"
# age: The person's age in years
#
#
# sex: The person's sex (1 = male, 0 = female)
#
#
# cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
#
#
# trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
#
#
# chol: The person's cholesterol measurement in mg/dl
#
#
# fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
#
#
# restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
#
#
# thalach: The person's maximum heart rate achieved
#
#
# exang: Exercise induced angina (1 = yes; 0 = no)
#
#
# oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
#
#
# slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
#
#
# ca: The number of major vessels (0-3)
#
#
# thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
#
#
# target: Heart disease (0 = no, 1 = yes)
#

# %% [markdown]
# <img src="https://mir-s3-cdn-cf.behance.net/project_modules/disp/19d6b946017221.584577bc3cd67.gif" width="700px">

# %% _kg_hide-input=true id="c3wP96Ub8bLp" outputId="5e048da9-6ae0-476f-f6e4-177f477ecd34"
# reading the head of the data

data.head()

# %% _kg_hide-input=true id="toeKaYyw8fgN" outputId="070cefdb-e9f2-481d-adf1-49f88357cdc6"
# describing the data

data.describe()

# %% [markdown]
# ## Data Profiling

# %% _kg_hide-input=true
profile = pandas_profiling.ProfileReport(data)
profile

# %% [markdown]
# ## Data Visualizations

# %% _kg_hide-input=true
import warnings
warnings.filterwarnings('ignore')

figure = bubbleplot(dataset = data, x_column = 'trestbps', y_column = 'chol', 
    bubble_column = 'sex', time_column = 'age', size_column = 'oldpeak', color_column = 'sex', 
    x_title = "Resting Blood Pressure", y_title = "Cholestrol", title = 'BP vs Chol. vs Age vs Sex vs Heart Rate',
    x_logscale = False, scale_bubble = 3, height = 650)

py.iplot(figure, config={'scrollzoom': True})

# %% _kg_hide-input=true id="z_kgWtZy-upN" outputId="6e75cf8f-b0b4-4c13-fe04-1e32a6092587"
# making a heat map

plt.rcParams['figure.figsize'] = (20, 15)
plt.style.use('ggplot')

sns.heatmap(data.corr(), annot = True, cmap = 'Wistia')
plt.title('Heatmap for the Dataset', fontsize = 20)
plt.show()

# %% [markdown]
# > The above heat map is to show the correlations amongst the different attributes of the given dataset. The above Heat Map shows that almost all of the features/attributes given in the dataset are very less correlated with each other. This implies we must include all of the features, as we can only eliminate those features where the correlation of two or more features are very high.

# %% _kg_hide-input=true id="5m8w6_Si8p_Y" outputId="898e1fa1-3232-47cd-ef25-3a596abe73f2"
# checking the distribution of age amonng the patients

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 5)
sns.distplot(data['age'], color = 'cyan')
plt.title('Distribution of Age', fontsize = 20)
plt.show()

# %% [markdown]
# > The above Distribution plot shows the distribution of Age amongst all of the entries in the dataset about the heart patients. The Graph suggests that the highest number of people suffering from heart diseases are in the age group of 55-65 years. The patients in the age group 20-30 are very less likely to suffer from heart diseases.
# >> As we know that the number of people in the age group 65-80 has a very low population, hence distribution is also less. we might have to opt for other plots to investigate further and get some more intuitive results.

# %% _kg_hide-input=true id="FeSmjob2Czwb" outputId="13fee9ed-76c1-4cf6-ea74-fa97f77a0f60"
# plotting a donut chart for visualizing each of the recruitment channel's share

size = data['sex'].value_counts()
colors = ['lightblue', 'lightgreen']
labels = "Male", "Female"
explode = [0, 0.01]

my_circle = plt.Circle((0, 0), 0.7, color = 'white')

plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%')
plt.title('Distribution of Gender', fontsize = 20)
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.legend()
plt.show()

# %% [markdown]
# > The above Pie chart, whhich shows us the distribution of Gender in the Heart diseases. By looking at the plot, we can **assume** that Males are two times more likely to suffer from heart diseases in comparison to females.
# >> According to our study, From all the Entries in our dataset 68% of the patients are men whereas only 32% are women. More number of men took participation in heart disease check ups.

# %% _kg_hide-input=true id="w-6sW63_DWhW" outputId="82c1fa59-44a6-4e00-9343-685d062db040"
# plotting the target attribute

plt.rcParams['figure.figsize'] = (15, 7)
plt.style.use('seaborn-talk')
sns.countplot(data['target'], palette = 'pastel')
plt.title('Distribution of Target', fontsize = 20)
plt.show()

# %% [markdown]
# > Let's look at the Target, The dataset is quite balanced with almost equal no. of Positive and Negative Classes. Let's say the Positive Class says that the patient is suffering from the disease and the Negative class says that the patient is not suffering from the disease.

# %% _kg_hide-input=true id="CSUo0OyHDoqC" outputId="4853cd1c-1690-4340-befd-3ce81e2f7792"
# tresbps vs target

plt.rcParams['figure.figsize'] = (12, 9)
sns.boxplot(data['target'], data['trestbps'], palette = 'viridis')
plt.title('Relation of tresbps with target', fontsize = 20)
plt.show()

# %% [markdown]
# > tresbps: Resting Blood Pressure, The above Bivariate plot between tresbps(the resting blood pressure of a patient), and the target which says that whether the patient is suffering from the heart disease or not. The plot clearly suggests that the patients who are most likely to not suffer from the disease have a slighly greater blood pressure than the patients who have heart diseases.

# %% _kg_hide-input=true id="vAl7BkvzEPkd" outputId="140e72a9-a6aa-4634-c58e-48baa9a4bf05"
# cholestrol vs target

plt.rcParams['figure.figsize'] = (12, 9)
sns.violinplot(data['target'], data['chol'], palette = 'colorblind')
plt.title('Relation of Cholestrol with Target', fontsize = 20, fontweight = 30)
plt.show()

# %% [markdown]
# > The above Bivariate plot between cholestrol levels and target suggests that the Patients likely to suffer from heart diseases are having higher cholestrol levels in comparison to the patients with target 0(likely to not suffer from the heart diseases.
# >> Hence, we can infer from the above plot that the cholestrol levels plays an important role in determining heart diseases. We all must keep our cholestrol levels in control as possible.

# %% _kg_hide-input=true id="chtJX61vEnF0" outputId="c6a23933-3e15-47cc-fb77-6b68b558307c"
# Resting electrocardiographic measurement vs target
  
plt.rcParams['figure.figsize'] = (12, 9)
dat = pd.crosstab(data['target'], data['restecg']) 
dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar', 
                                                 stacked = False, 
                                                 color = plt.cm.rainbow(np.linspace(0, 1, 4)))
plt.title('Relation of ECG measurement with Target', fontsize = 20, fontweight = 30)
plt.show()

# %% [markdown]
# > The above plot is column bar chart representing target vs ECG Measurements(Electro Cardio Gram), The above plot shows that the more number of patients not likely to suffer from heart diseases are having restscg value 0 whereas more number of people have restecg value 1 in case of more likelihood of suffering from a heart disease.

# %% [markdown]
# > This Heat Map, between Target and Maximum Heart Rate shows that the patients who are likely to suffer from heart diseases are having higher maximum heart rates whereas the patients who are not likely to suffer from any heart diseases are having lower maximum heart rates.
# >> This implies it is very important to keep our heart rates low, to keep ourselves healthy and safe from any dangerous heart diseases.

# %% _kg_hide-input=true id="H80e5avXHj5D" outputId="6d0f8e70-ba2d-4ad7-9438-9c603ddb9948"
# slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# checking the relation between slope and target

plt.rcParams['figure.figsize'] = (15, 9)
sns.boxenplot(data['target'], data['slope'], palette = 'copper')
plt.title('Relation between Peak Exercise and Target', fontsize = 20, fontweight = 30)
plt.show()

# %% [markdown]
# > Slope : 0 refers to upsloping, 1 refers to flat Exercises pattern.
# >>This plot clearly shows that the patients who are not likely to suffer from any heart diseases are mostly having value 1 means upsloping, whereas very few people suffering from heart diseases have upsloping pattern in exercises.
# >> Also, Flat Exercises are mostly seen in the cases of Patients who are more likely to suffer from heart diseases.

# %% _kg_hide-input=true id="XH4TyI35HtFx" outputId="bce3803f-01a6-4d54-8aa5-57e9ac17e73a"
#ca: The number of major vessels (0-3)

sns.boxenplot(data['target'], data['ca'], palette = 'Reds')
plt.title('Relation between no. of major vessels and target', fontsize = 20, fontweight = 30)
plt.show()

# %% [markdown]
# > The above Bivariate plot between Target and Number of Major Vessels, shows that the patients who are more likely to suffer from Heart diseases are having high values of Major Vessels wheras the patiets who are very less likely to suffer from any kind of heart diseases have very low values of Major Vessels.
# >> Hence, It is also helpful in determining the heart diseases, the more the number of vessels, the more is the chance of suffering from heart diseases.

# %% _kg_hide-input=true id="u8q2vLcnHs_w" outputId="15af2985-04af-4179-edd4-6d8763ba2f2f"
# relation between age and target

plt.rcParams['figure.figsize'] = (15, 9)
sns.swarmplot(data['target'], data['age'], palette = 'winter', size = 10)
plt.title('Relation of Age and target', fontsize = 20, fontweight = 30)
plt.show()

# %% [markdown]
# > From the above Swarm plot between the target and the age of the patients, we are not able to find any clue or pattern, so age is not a very good attribute to determine the heart disease of a patient as a patient of heart diseases range from 30-70, whereas it is not important that all of the people lying in that same age group are bound to suffer from the heart diseases.

# %% _kg_hide-input=true id="4s9NyXwfJCB7" outputId="6e10eaf5-3ffd-4aa8-b267-2b3caec51b56"
# relation between sex and target

sns.boxenplot(data['target'], data['sex'], palette = 'Set3')
plt.title('Relation of Sex and target', fontsize = 20, fontweight = 30)
plt.show()

# %% _kg_hide-input=true id="TID8qXc0JnIF" outputId="9f3a7696-da77-4fa6-9200-5afb0743430f"
# checking the relation between 
#thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)

sns.boxenplot(data['target'], data['thal'], palette = 'magma')
plt.title('Relation between Target and Blood disorder-Thalessemia', fontsize = 20, fontweight = 30)
plt.show()

# %% [markdown]
# >In the above Boxen plot between Target and a Blood disorder called Thalessemia, It can be easily inferred that the patients suffering from heart diseases have low chances of also suffering from thalessemia in comparison to the patients who are less likely to suffer from the heart diseases. Hence, It is also a good feature to classify heart diseases.

# %% _kg_hide-input=true id="qBGaq9zCKge7" outputId="24355f4d-67c0-4948-87d8-042688af051a"
# target vs chol and hue = thalach

plt.rcParams['figure.figsize'] = (15, 8)
plt.style.use('fivethirtyeight')
plt.scatter(x = data['target'], y = data['chol'], s = data['thalach']*100, color = 'yellow')
plt.title('Relation of target with cholestrol and thalessemia', fontsize = 20, fontweight = 30)
plt.show()

# %% _kg_hide-input=true id="wPUkSpR3LLD7" outputId="a7147426-2c48-4239-bfe3-0ff14adc260d"
# multi-variate analysis

sns.boxplot(x = data['target'], y = data['trestbps'], hue = data['sex'], palette = 'rainbow')
plt.title('Checking relation of tresbps with genders to target', fontsize = 20, fontweight = 30)
plt.show()

# %% [markdown]
# > In the above Box plot between Target and tresbps wrt Gender, shows that Women have higher tresbps than men in case of not suffering from any heart diseases, whereas men and women have almost equal tresbps in case of suffering from a heart diseases. Also, In case of suffering from heart diseases, patients have a slightly lower tresbps in comparison to the patients who are not suffering from heart diseases.

# %% _kg_hide-input=true _kg_hide-output=false
trace = go.Scatter3d(
    x = data['chol'],
    y = data['trestbps'],
    z = data['age'],
    name = 'Marvel',
    mode = 'markers',
    marker = dict(
         size = 10,
         color = data['age']
    )
)

df = [trace]

layout = go.Layout(
    title = 'Cholestrol vs Heart Rate vs Age',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    ),
    scene = dict(
            xaxis = dict(title  = 'Cholestrol'),
            yaxis = dict(title  = 'Heart Rate'),
            zaxis = dict(title  = 'Age')
        )
    
)
fig = go.Figure(data = df, layout=layout)
py.iplot(fig)

# %% _kg_hide-output=true id="v9RUg4_n80k5" outputId="0cc3839f-0922-481e-85d3-b33edbf488c9"
# let's change the names of the  columns for better understanding

data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

data.columns

# %% _kg_hide-output=true id="JkA3_NBf_CnJ" outputId="913a92cd-1989-4347-a30b-67d88a7b5b22"
data['sex'][data['sex'] == 0] = 'female'
data['sex'][data['sex'] == 1] = 'male'

data['chest_pain_type'][data['chest_pain_type'] == 1] = 'typical angina'
data['chest_pain_type'][data['chest_pain_type'] == 2] = 'atypical angina'
data['chest_pain_type'][data['chest_pain_type'] == 3] = 'non-anginal pain'
data['chest_pain_type'][data['chest_pain_type'] == 4] = 'asymptomatic'

data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

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
data['exercise_induced_angina'] = data['exercise_induced_angina'].astype('object')
data['st_slope'] = data['st_slope'].astype('object')
data['thalassemia'] = data['thalassemia'].astype('object')

# %% id="NQCiGUCudfOA" outputId="63d07d19-d863-4907-d48e-34e61b06ae7b"
# taking the labels out from the data

y = data['target']

data = data.drop('target', axis = 1)

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

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

model = RandomForestClassifier(n_estimators = 50, max_depth = 5)
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
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')

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


export_graphviz(estimator, out_file='tree.dot', 
                feature_names = feature_names,
                class_names = y_train_str,
                rounded = True, proportion = True, 
                label='root',
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=50'])

from IPython.display import Image
Image(filename = 'tree.png')


# %% [markdown] id="M7hZyrUnVIDw"
# **Specificity and Sensitivity**

# %% id="9hNFsTsPShN6" outputId="91c55840-76a3-48d1-e39a-4401fa62d47a"
total=sum(sum(cm))

sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])
print('Sensitivity : ', sensitivity )

specificity = cm[1,1]/(cm[1,1]+cm[0,1])
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
plt.title('ROC curve for diabetes classifier', fontweight = 30)
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
from pdpbox import pdp, info_plots #for partial plots


# %% [markdown]
# **Eli5 Values**

# %% _kg_hide-input=true id="arM0hk6CYNRy" outputId="c9c0cf83-694e-42d9-c63e-a6a255f3b66a"
# let's check the importance of each attributes

perm = PermutationImportance(model, random_state = 0).fit(x_test, y_test)
eli5.show_weights(perm, feature_names = x_test.columns.tolist())

# %% [markdown] id="7GizL0WEcFbr"
# **Partial Dependence Plot for Top 5 Features**

# %% _kg_hide-input=true id="Vwoe1xOKbFBL" outputId="e0615292-a163-4adf-f0d9-96e81bc7ed87"
# plotting the partial dependence plot for num_major_vessels

base_features = data.columns.values.tolist()

feat_name = 'num_major_vessels'
pdp_dist = pdp.pdp_isolate(model=model, dataset=x_test, model_features = base_features, feature = feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# %% _kg_hide-input=true id="DVOfQUXrcRzO" outputId="2db8f7d6-46d9-4330-f819-8f96cd8547a1"
# let's plot the partial dependence plot for thalassemia_fixed defect

base_features = data.columns.values.tolist()

feat_name = 'thalassemia_fixed defect'
pdp_dist = pdp.pdp_isolate(model = model, dataset = x_test, model_features = base_features, feature = feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# %% _kg_hide-input=true id="TnEBkXd0hN2x" outputId="882b6c6d-d926-4a29-c3b8-1d449dfacbcf"

# let's plot the partial dependence plot for thalassemia_reversable defect

base_features = data.columns.values.tolist()

feat_name = 'thalassemia_reversable defect'
pdp_dist = pdp.pdp_isolate(model = model, dataset = x_test, model_features = base_features, feature = feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

# %% _kg_hide-input=true id="iw5qC8FbhcxN" outputId="51397baa-d1e9-40b2-8c77-ee8ccd952733"
# plotting a partial dependence graph for chest_pain_type_atypical angina

base_features = data.columns.values.tolist()

feat_name = 'chest_pain_type_atypical angina'
pdp_dist = pdp.pdp_isolate(model = model, dataset = x_test, model_features = base_features, feature = feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# %% _kg_hide-input=true id="qvrDYACPh0qr" outputId="04bdbef1-7e79-4b46-ae72-7e6e81587c7b"
# plotting a partial dependence graph for st_depression


base_features = data.columns.values.tolist()

feat_name = 'st_depression'
pdp_dist = pdp.pdp_isolate(model = model, dataset = x_test, model_features = base_features, feature = feat_name)

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
  return shap.force_plot(explainer.expected_value[1], shap_values[1], patient)




# %% [markdown] id="hUL8TpYPr2iX"
# **Report for the First Patient**

# %% _kg_hide-input=true id="S8fgkfytrS5T" outputId="6bfc1189-fb5a-4945-8f05-4d5dff95509a"
# let's do some real time prediction for patients

patients = x_test.iloc[1,:].astype(float)
patient_analysis(model, patients)



# %% [markdown] id="LbLQnBEfr6ez"
# **Report for the Second Patient**

# %% _kg_hide-input=true id="u0C_dGsdry5O" outputId="d335ad96-0bac-4257-fa15-a8890f968a0e"
patients = x_test.iloc[:, 2].astype(float)
patient_analysis(model, patients)



# %% [markdown] id="xa4ebhIIsKYp"
# **Report for the Third Patient**

# %% _kg_hide-input=true id="UW5BCyV8sGh0" outputId="f7d90392-7de6-4a35-b0d9-c6dcf6cdb52e"
patients = x_test.iloc[:,3].astype(float)
patient_analysis(model, patients)



# %% id="v-mjo5JnsW-Z" outputId="47d8313c-9535-4740-995c-9999c213b24e"
# dependence plot

shap.dependence_plot('num_major_vessels', shap_values[1], x_test, interaction_index = "st_depression")



# %% [markdown] id="Ye8I2f1Rs-dW"
# **Force Plot**

# %% id="sRgtSI4Js5Uc" outputId="a9413b7d-0273-40ee-a708-e10fa14c3eb7"
shap_values = explainer.shap_values(x_train.iloc[:50])
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], x_test.iloc[:50])


# %%











































































