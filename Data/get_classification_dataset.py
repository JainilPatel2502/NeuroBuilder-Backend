from torch.utils.data import Dataset
import torch
import pandas as pd
class MyClassificationDataset(Dataset):
    def __init__(self, df):
        self.df = df
        # Coerce all columns to numeric — catches bool/object dtype from one-hot CSV round-trips
        X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0).values
        Y = df.iloc[:, -1].apply(pd.to_numeric, errors="coerce").fillna(0).values
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
