import pandas as pd

def split_by_date(df, cutoff='2025-01-01'):

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    cutoff = pd.Timestamp(cutoff)

    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]

    return train, test