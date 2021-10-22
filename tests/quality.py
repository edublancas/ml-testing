from pathlib import Path
import pickle

from scipy.stats import ks_2samp
from sklearn.metrics import f1_score
import pandas as pd


def clean(product):
    df = pd.read_csv(product['data'])

    assert df.age.min() > 0
    assert set(df.sex.unique()) == {'female', 'male'}

    # check if the distribution of age is the same
    ref = pd.read_csv('reference/clean.csv')
    # # https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    same_distribution = ks_2samp(df.age, ref.age).pvalue > 0.05

    # raise an error if it has changed
    assert same_distribution


def fit(product):
    y_pred = pickle.loads(Path(product['y_pred']).read_bytes())
    y_test = pickle.loads(Path(product['y_test']).read_bytes()).values

    # important to check both sides! a suddently "good" model is also bad
    # news and should be verified
    score = f1_score(y_test, y_pred)
    assert 0.86 <= score <= 0.92, f'unexpected f1 score: {score}'