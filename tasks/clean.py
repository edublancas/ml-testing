import transform
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

# %%
# NOTE: values reported in the original source (UCI repository) are from 1-4
# but this copy I downloaded from Kaggle has values 0-3, so I'm assuming
# they're just shifted
# we abstract the transformation so we can unit test it (see tests/test_transform.py)
data['chest_pain_type'] = transform.chest_pain_type(data['chest_pain_type'])

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

data.to_csv(product['data'], index=False)