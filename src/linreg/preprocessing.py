import pandas as pd

def split_by_date(df, cutoff='2025-01-01'):

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    cutoff = pd.Timestamp(cutoff)

    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]

    return train, test

CATEGORICAL_COLS = ['day_of_week', 'month', 'is_holiday']

def fit_one_hot(train, cols=CATEGORICAL_COLS):
    return {col: sorted(train[col].dropna().unique()) for col in cols}

def transform_one_hot(df, categories):
    df = df.copy()
    for col, cats in categories.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    return pd.get_dummies(df, columns=list(categories), drop_first=True, dtype=int)
