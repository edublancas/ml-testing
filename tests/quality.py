import pandas as pd

def clean(product):
    df = pd.read_csv(product['data'])

    assert df.age.min() > 0
    assert set(df.sex.unique()) == {'female', 'male'}
