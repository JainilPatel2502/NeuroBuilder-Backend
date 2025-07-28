import torch.nn as nn
from ModelBuilder.Model import Model as ModelClass
def layer_fromater(input,layer):
    formatted_layer = []
    prev = input
    for neuron in layer:
        lay = (prev,neuron)
        prev = neuron
        formatted_layer.append(lay)
    return formatted_layer

def create_layers(no_of_layers, formatted_layer, activations, initializations):
    layers = []
    for i in range(no_of_layers):
        in_features, out_features = formatted_layer[i]
        layer = nn.Linear(in_features=in_features, out_features=out_features)
        
        if initializations[i] == "he" or initializations[i] == "Normalized":
            nonlinearity = 'relu' if activations[i] in ["ReLU", "PReLU"] else 'leaky_relu' if activations[i] == "LeakyReLU" else 'relu'
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity=nonlinearity)
        elif initializations[i] == "xavier":
            nn.init.xavier_normal_(layer.weight)
        else:
            print(f"Warning: Unknown initialization '{initializations[i]}' at layer {i}, using default initialization.")
        
        layers.append(layer)
        if activations[i] == "ReLU":
            layers.append(nn.ReLU())
        elif activations[i] == "Sigmoid":
            layers.append(nn.Sigmoid())
        elif activations[i] == "PReLU":
            layers.append(nn.PReLU())
        elif activations[i] == "ELU":
            layers.append(nn.ELU())
        elif activations[i] == "TanH":
            layers.append(nn.Tanh())
        elif activations[i] == "Softmax":
            layers.append(nn.Softmax(dim=1))
        elif activations[i] == "LeakyReLU":
            layers.append(nn.LeakyReLU())
        elif activations[i]=='None':
            continue
        else:
            print(f"Warning: Unknown activation '{activations[i]}' at layer {i}, skipping activation.")
    return layers


def model_builder(input_size,data):
    formated =  layer_fromater(input_size,data['neuronsPerLayer'])
    layers = create_layers(data['layers'],formated,data['actiavtionsPerLayer'],data['initializationPerLayer'])
    model = ModelClass(layers)
    return  model


# data = {
#     "layers": 4,
#     "neuronsPerLayer": [
#         7,
#         4,
#         3,
#         1
#     ],
#     "actiavtionsPerLayer": [
#         "PReLU",
#         "ReLU",
#         "TanH",
#         "None"
#     ],
#     "initializationPerLayer": [
#         "Normalized",
#         "Normalized",
#         "Normalized",
#         "Normalized"
#     ],
#     "regularization": "L2",
#     "lr": 0.01,
#     "lossFn": "MAE",
#     "optimzer": "Adam",
#     "epochs": 100
# }
# print(model_builder(data))
