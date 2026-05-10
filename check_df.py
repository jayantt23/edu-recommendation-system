import pandas as pd
import numpy as np

DATA_PATH = "data/processed/final_hybrid_dataset.pkl"
df = pd.read_pickle(DATA_PATH)
print(df.head())
print(df.dtypes)
print(df['ncessch'].iloc[0])
