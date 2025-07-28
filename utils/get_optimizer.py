import torch.optim as optim
def get_optimizer(params,name:str,lr:float):
    if name =="SGD":
        return optim.SGD(params,lr=lr)
    elif name=="RMS Prop":
        return optim.RMSprop(params,lr=lr)
    elif name=="Adam":
        return optim.Adam(params,lr=lr)
    elif name=="AdamW":
        return optim.AdamW(params,lr=lr)
    elif name=="Adadelta":
        return optim.Adadelta(params,lr=lr)
