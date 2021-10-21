from scipy.stats import ks_2samp
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
