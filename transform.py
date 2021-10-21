"""
Functions for transforming raw data
"""


def chest_pain_type(series):
    """Convert chest_pain_type series from integer to string
    """
    return series.map({
        0: 'typical angina',
        1: 'atypical angina',
        2: 'non-anginal pain',
        3: 'asymptomatic'
    })
