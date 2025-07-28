import torch.nn as nn
def get_loss_fn(name:str):
    if name == "Huber":
        loss = nn.HuberLoss()
        return loss
    elif name =="MSE":
        return nn.MSELoss()
    elif name =="MAE":
        return nn.L1Loss()
    elif name =="Categorical Crossentropy":
        return nn.CrossEntropyLoss()
    elif name =="Binary Cross Entropy":
        return nn.BCELoss()