from abc import *

import torch
import torch.nn as nn
class NetworkBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        super(NetworkBase, self).__init__()
    @abstractmethod
    def forward(self, x):
        return x

from abc import *

import torch
import torch.nn as nn

class NetworkBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        super(NetworkBase, self).__init__()

    @abstractmethod
    def forward(self, x):
        return x

class Network(NetworkBase):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function=torch.relu, last_activation=None, use_layernorm=False):
        super(Network, self).__init__()
        self.activation = activation_function
        self.last_activation = last_activation
        self.use_layernorm = use_layernorm

        layers_unit = [input_dim] + [hidden_dim] * (layer_num - 1)
        self.layers = nn.ModuleList()
        self.layernorms = nn.ModuleList()

        for idx in range(len(layers_unit) - 1):
            self.layers.append(nn.Linear(layers_unit[idx], layers_unit[idx + 1]))
            if self.use_layernorm:
                self.layernorms.append(nn.LayerNorm(layers_unit[idx + 1]))

        self.last_layer = nn.Linear(layers_unit[-1], output_dim)
        self.network_init()

    def forward(self, x):
        return self._forward(x)

    def _forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_layernorm:
                x = self.layernorms[i](x)
            x = self.activation(x)
        x = self.last_layer(x)
        if self.last_activation is not None:
            x = self.last_activation(x)
        return x

    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()
#
# class Network(NetworkBase):
#     def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.relu,last_activation = None):
#         super(Network, self).__init__()
#         self.activation = activation_function
#         self.last_activation = last_activation
#         layers_unit = [input_dim]+ [hidden_dim]*(layer_num-1)
#         layers = ([nn.Linear(layers_unit[idx],layers_unit[idx+1]) for idx in range(len(layers_unit)-1)])
#         self.layers = nn.ModuleList(layers)
#         self.last_layer = nn.Linear(layers_unit[-1],output_dim)
#         self.network_init()
#     def forward(self, x):
#         return self._forward(x)
#     def _forward(self, x):
#         for layer in self.layers:
#             x = self.activation(layer(x))
#         x = self.last_layer(x)
#         if self.last_activation != None:
#             x = self.last_activation(x)
#         return x
#     def network_init(self):
#         for layer in self.modules():
#             if isinstance(layer, nn.Linear):
#                 nn.init.orthogonal_(layer.weight)
#                 layer.bias.data.zero_()