from torch.utils.data import Dataset
import torch
class MyRegressionDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.x=  torch.tensor(df.iloc[:,:-1].values,dtype=torch.float32)  
        self.y = torch.tensor(df.iloc[:,-1].values,dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
