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
init_notebook_mode(connected=True)
from bubbly.bubbly import bubbleplot

# for model explanation
import shap

# %% tags=["parameters"]
upstream = None
product = None
path_to_data = None

# %% _kg_hide-input=true _kg_hide-output=true id="0No8Wdtb8Mlw" outputId="b883289b-5690-4abb-bd4d-0cfc3fab5fa8"
# reading the data
data = pd.read_csv(path_to_data)

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

figure = bubbleplot(dataset=data,
                    x_column='trestbps',
                    y_column='chol',
                    bubble_column='sex',
                    time_column='age',
                    size_column='oldpeak',
                    color_column='sex',
                    x_title="Resting Blood Pressure",
                    y_title="Cholestrol",
                    title='BP vs Chol. vs Age vs Sex vs Heart Rate',
                    x_logscale=False,
                    scale_bubble=3,
                    height=650)

py.iplot(figure, config={'scrollzoom': True})

# %% _kg_hide-input=true id="z_kgWtZy-upN" outputId="6e75cf8f-b0b4-4c13-fe04-1e32a6092587"
# making a heat map

plt.rcParams['figure.figsize'] = (20, 15)
plt.style.use('ggplot')

sns.heatmap(data.corr(), annot=True, cmap='Wistia')
plt.title('Heatmap for the Dataset', fontsize=20)
plt.show()

# %% [markdown]
# > The above heat map is to show the correlations amongst the different attributes of the given dataset. The above Heat Map shows that almost all of the features/attributes given in the dataset are very less correlated with each other. This implies we must include all of the features, as we can only eliminate those features where the correlation of two or more features are very high.

# %% _kg_hide-input=true id="5m8w6_Si8p_Y" outputId="898e1fa1-3232-47cd-ef25-3a596abe73f2"
# checking the distribution of age amonng the patients

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 5)
sns.distplot(data['age'], color='cyan')
plt.title('Distribution of Age', fontsize=20)
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

my_circle = plt.Circle((0, 0), 0.7, color='white')

plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(size,
        colors=colors,
        labels=labels,
        shadow=True,
        explode=explode,
        autopct='%.2f%%')
plt.title('Distribution of Gender', fontsize=20)
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
sns.countplot(data['target'], palette='pastel')
plt.title('Distribution of Target', fontsize=20)
plt.show()

# %% [markdown]
# > Let's look at the Target, The dataset is quite balanced with almost equal no. of Positive and Negative Classes. Let's say the Positive Class says that the patient is suffering from the disease and the Negative class says that the patient is not suffering from the disease.

# %% _kg_hide-input=true id="CSUo0OyHDoqC" outputId="4853cd1c-1690-4340-befd-3ce81e2f7792"
# tresbps vs target

plt.rcParams['figure.figsize'] = (12, 9)
sns.boxplot(data['target'], data['trestbps'], palette='viridis')
plt.title('Relation of tresbps with target', fontsize=20)
plt.show()

# %% [markdown]
# > tresbps: Resting Blood Pressure, The above Bivariate plot between tresbps(the resting blood pressure of a patient), and the target which says that whether the patient is suffering from the heart disease or not. The plot clearly suggests that the patients who are most likely to not suffer from the disease have a slighly greater blood pressure than the patients who have heart diseases.

# %% _kg_hide-input=true id="vAl7BkvzEPkd" outputId="140e72a9-a6aa-4634-c58e-48baa9a4bf05"
# cholestrol vs target

plt.rcParams['figure.figsize'] = (12, 9)
sns.violinplot(data['target'], data['chol'], palette='colorblind')
plt.title('Relation of Cholestrol with Target', fontsize=20, fontweight=30)
plt.show()

# %% [markdown]
# > The above Bivariate plot between cholestrol levels and target suggests that the Patients likely to suffer from heart diseases are having higher cholestrol levels in comparison to the patients with target 0(likely to not suffer from the heart diseases.
# >> Hence, we can infer from the above plot that the cholestrol levels plays an important role in determining heart diseases. We all must keep our cholestrol levels in control as possible.

# %% _kg_hide-input=true id="chtJX61vEnF0" outputId="c6a23933-3e15-47cc-fb77-6b68b558307c"
# Resting electrocardiographic measurement vs target

plt.rcParams['figure.figsize'] = (12, 9)
dat = pd.crosstab(data['target'], data['restecg'])
dat.div(dat.sum(1).astype(float),
        axis=0).plot(kind='bar',
                     stacked=False,
                     color=plt.cm.rainbow(np.linspace(0, 1, 4)))
plt.title('Relation of ECG measurement with Target',
          fontsize=20,
          fontweight=30)
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
sns.boxenplot(data['target'], data['slope'], palette='copper')
plt.title('Relation between Peak Exercise and Target',
          fontsize=20,
          fontweight=30)
plt.show()

# %% [markdown]
# > Slope : 0 refers to upsloping, 1 refers to flat Exercises pattern.
# >>This plot clearly shows that the patients who are not likely to suffer from any heart diseases are mostly having value 1 means upsloping, whereas very few people suffering from heart diseases have upsloping pattern in exercises.
# >> Also, Flat Exercises are mostly seen in the cases of Patients who are more likely to suffer from heart diseases.

# %% _kg_hide-input=true id="XH4TyI35HtFx" outputId="bce3803f-01a6-4d54-8aa5-57e9ac17e73a"
#ca: The number of major vessels (0-3)

sns.boxenplot(data['target'], data['ca'], palette='Reds')
plt.title('Relation between no. of major vessels and target',
          fontsize=20,
          fontweight=30)
plt.show()

# %% [markdown]
# > The above Bivariate plot between Target and Number of Major Vessels, shows that the patients who are more likely to suffer from Heart diseases are having high values of Major Vessels wheras the patiets who are very less likely to suffer from any kind of heart diseases have very low values of Major Vessels.
# >> Hence, It is also helpful in determining the heart diseases, the more the number of vessels, the more is the chance of suffering from heart diseases.

# %% _kg_hide-input=true id="u8q2vLcnHs_w" outputId="15af2985-04af-4179-edd4-6d8763ba2f2f"
# relation between age and target

plt.rcParams['figure.figsize'] = (15, 9)
sns.swarmplot(data['target'], data['age'], palette='winter', size=10)
plt.title('Relation of Age and target', fontsize=20, fontweight=30)
plt.show()

# %% [markdown]
# > From the above Swarm plot between the target and the age of the patients, we are not able to find any clue or pattern, so age is not a very good attribute to determine the heart disease of a patient as a patient of heart diseases range from 30-70, whereas it is not important that all of the people lying in that same age group are bound to suffer from the heart diseases.

# %% _kg_hide-input=true id="4s9NyXwfJCB7" outputId="6e10eaf5-3ffd-4aa8-b267-2b3caec51b56"
# relation between sex and target

sns.boxenplot(data['target'], data['sex'], palette='Set3')
plt.title('Relation of Sex and target', fontsize=20, fontweight=30)
plt.show()

# %% _kg_hide-input=true id="TID8qXc0JnIF" outputId="9f3a7696-da77-4fa6-9200-5afb0743430f"
# checking the relation between
#thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)

sns.boxenplot(data['target'], data['thal'], palette='magma')
plt.title('Relation between Target and Blood disorder-Thalessemia',
          fontsize=20,
          fontweight=30)
plt.show()

# %% [markdown]
# >In the above Boxen plot between Target and a Blood disorder called Thalessemia, It can be easily inferred that the patients suffering from heart diseases have low chances of also suffering from thalessemia in comparison to the patients who are less likely to suffer from the heart diseases. Hence, It is also a good feature to classify heart diseases.

# %% _kg_hide-input=true id="qBGaq9zCKge7" outputId="24355f4d-67c0-4948-87d8-042688af051a"
# target vs chol and hue = thalach

plt.rcParams['figure.figsize'] = (15, 8)
plt.style.use('fivethirtyeight')
plt.scatter(x=data['target'],
            y=data['chol'],
            s=data['thalach'] * 100,
            color='yellow')
plt.title('Relation of target with cholestrol and thalessemia',
          fontsize=20,
          fontweight=30)
plt.show()

# %% _kg_hide-input=true id="wPUkSpR3LLD7" outputId="a7147426-2c48-4239-bfe3-0ff14adc260d"
# multi-variate analysis

sns.boxplot(x=data['target'],
            y=data['trestbps'],
            hue=data['sex'],
            palette='rainbow')
plt.title('Checking relation of tresbps with genders to target',
          fontsize=20,
          fontweight=30)
plt.show()

# %% [markdown]
# > In the above Box plot between Target and tresbps wrt Gender, shows that Women have higher tresbps than men in case of not suffering from any heart diseases, whereas men and women have almost equal tresbps in case of suffering from a heart diseases. Also, In case of suffering from heart diseases, patients have a slightly lower tresbps in comparison to the patients who are not suffering from heart diseases.

# %% _kg_hide-input=true _kg_hide-output=false
trace = go.Scatter3d(x=data['chol'],
                     y=data['trestbps'],
                     z=data['age'],
                     name='Marvel',
                     mode='markers',
                     marker=dict(size=10, color=data['age']))

df = [trace]

layout = go.Layout(title='Cholestrol vs Heart Rate vs Age',
                   margin=dict(l=0, r=0, b=0, t=0),
                   scene=dict(xaxis=dict(title='Cholestrol'),
                              yaxis=dict(title='Heart Rate'),
                              zaxis=dict(title='Age')))
fig = go.Figure(data=df, layout=layout)
py.iplot(fig)
