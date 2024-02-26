# Software Name : carla_croco_vcnet
# Version: 1.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# the text of which is available at https://spdx.org/licenses/MIT.html
# or see the "license.txt" file for more details.


import torch.nn as nn
import torch 
from carla.self_explaining_model.catalog.vcnet.library.vcnet_tabular_data_v0.join_training_network import Predictor
from typing import Union
# Transform sigmoid output vector into a softmax format probability vector 
class sigmoid_to_ohe(nn.Module) : 
    def __init__(self):
        super().__init__()
    
    def forward(self,input) : 
        return(torch.hstack([1-input,input]))


# Load classif_model_part of VCnet as a pytorch model
### Include input of size 2 for compatibility with the framework 
def load_classif_model(training,immutable=False) : 

    if immutable :
        from carla.self_explaining_model.catalog.vcnet.library.vcnet_tabular_data_v1.join_training_network import Predictor
    else : 
        from carla.self_explaining_model.catalog.vcnet.library.vcnet_tabular_data_v0.join_training_network import Predictor

    # Predictor part of Vcnet as pytorch model 
    prediction_model = Predictor(**training.model.kwargs)
    # Load weights from Vcnet architecture 
    state_dict = prediction_model.state_dict().copy()
    with torch.no_grad():
        for layer in training.model.state_dict():
            if layer in state_dict : 
                state_dict[layer] = training.model.state_dict()[layer]

    prediction_model.load_state_dict(state_dict)
    
    # Change sigmoid output to softmax output 
    layers = [] 
    layers.append(prediction_model)
    layers.append(sigmoid_to_ohe())

    classif_model = nn.Sequential(*layers)
    return(classif_model)
