# Software Name : carla_croco_vcnet
# Version: 1.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# the text of which is available at https://spdx.org/licenses/MIT.html
# or see the "license.txt" file for more details.


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:09:29 2022

@author: nwgl2572
"""

from carla.self_explaining_model.catalog.vcnet.library.utils import *
from torch import nn
from sklearn.preprocessing import StandardScaler,MinMaxScaler, OneHotEncoder
import pathlib 

# Function that create a dictionary that map features to index 
def cat_index_ohe(cat_arrays : list ,continous_cols : list ,discret_cols : list) : 
    dico = {} 
    cat_idx = len(continous_cols)
    for i in range(len(continous_cols)) : 
        dico[continous_cols[i]] = [i,i+1]
    for i in range (len(cat_arrays)):
        cat_end_idx = cat_idx + len(cat_arrays[i])
        cat_col = discret_cols[i] 
        index = [cat_idx,cat_end_idx] 
        dico[cat_col] = index
        cat_idx = cat_end_idx
    return(dico)

# Function that split mutable and immutable variables       
def split_immutable(X : torch.tensor ,cat_arrays : list ,continous_cols : list ,discret_cols : list ,immutable_features: list) : 
    dico = cat_index_ohe(cat_arrays,continous_cols,discret_cols)
    cont_shape_mutable = 0
    cat_arrays_mutable = []
    if immutable_features is not None : 
        X_immutable = []
        X_mutable = []
        for e in dico.keys() : 
            if e in immutable_features : 
                X_immutable.append(X[:,dico[e][0]:dico[e][1]])
            elif e not in immutable_features : 
                if e in continous_cols : 
                    cont_shape_mutable +=1
                elif e in discret_cols : 
                    cat_arrays_mutable.append(cat_arrays[discret_cols.index(e)])                        
                X_mutable.append(X[:,dico[e][0]:dico[e][1]])
    
        return(torch.hstack(X_immutable),torch.hstack(X_mutable),cont_shape_mutable,cat_arrays_mutable)
        