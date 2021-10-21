import pandas as pd
import transform


def test_transform_chest_pain_type():
    series = pd.Series([0, 1, 2, 3])
    expected = pd.Series([
        'typical angina',
        'atypical angina',
        'non-anginal pain',
        'asymptomatic',
    ])
    assert transform.chest_pain_type(series).equals(expected)


# tests for other transformations go here...