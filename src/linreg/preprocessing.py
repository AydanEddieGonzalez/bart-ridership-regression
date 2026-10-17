import pandas as pd

CATEGORICAL_COLS = ['day_of_week', 'month', 'station']
NUMERICAL_COLS =['temp_max_f', 'precip_in']

def split_by_date(df, cutoff='2025-01-01'):

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    cutoff = pd.Timestamp(cutoff)

    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]

    return train, test



def fit_one_hot(train, cols=CATEGORICAL_COLS):
    return {col: sorted(train[col].dropna().unique()) for col in cols}

def transform_one_hot(df, categories):
    df = df.copy()
    for col, cats in categories.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    return pd.get_dummies(df, columns=list(categories), drop_first=True, dtype=int)


def fit_scaler(train, cols=NUMERICAL_COLS):
    stats = {}
    for col in cols:
        mean = train[col].mean()
        std = train[col].std()
        if std == 0:
            std = 1.0
        stats[col] = (mean, std)
    return stats

def transform_scaler(df, stats):
    df = df.copy()
    for col, (mean, std) in stats.items():
        df[col] = (df[col] - mean) / std
    return df