import pandas as pd
from numpy import random


def random_rample_between(a, b, n):
    return (b - a) * random.random_sample(n) + a


def sample(n, include_target=False):
    """
    Sample synthetic data to simulate new observations
    """
    df = pd.DataFrame({
        'age': random.random_sample(n) * 80,
        'sex': random.choice([0, 1], n),
        'cp': random.choice([0, 1, 2, 3], n),
        'trestbps': random_rample_between(94, 200, n),
        'chol': random_rample_between(126, 564, n),
        'fbs': random.choice([0, 1], n),
        'restecg': random.choice([0, 1, 2], n),
        'thalach': random_rample_between(71, 202, n),
        'exang': random.choice([0, 1], n),
        'oldpeak': random_rample_between(0, 6.2, n),
        'slope': random.choice([0, 1, 2], n),
        'ca': random.choice([0, 1, 2, 3, 4], n),
        'thal': random.choice([0, 1, 2, 3], n),
    })

    if include_target:
        df['target'] = random.choice([0, 1], n)

    return df