from Data.get_classification_dataset import MyClassificationDataset
from Data.get_regression_dataset import MyRegressionDataset
from Data.get_splited import get_split
import pandas as pd
from torch.utils.data import DataLoader
class Datahandler:
    def __init__(self,projname:str , type , split,batch_size):
        self.df=pd.read_csv(f'./Projects/{projname}.csv')
        self.train, self.test = get_split(self.df,split)  
        if type=='classification':
            self.traindataset = MyClassificationDataset(self.train)
            self.testdataset = MyClassificationDataset(self.test)
        if type=='regression':
            self.traindataset = MyRegressionDataset(self.train)
            self.testdataset = MyRegressionDataset(self.test)
        self.trainloader = DataLoader(self.traindataset , batch_size=batch_size)
        self.testloader = DataLoader(self.testdataset , batch_size=batch_size)



# obj = Datahandler('hi','regression',0.2)
# print(obj.traindataset)
# for x, y  in obj.testdataset:
#     print(x.shape)
#     break