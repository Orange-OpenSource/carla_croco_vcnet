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

from carla.self_explaining_model.catalog.vcnet.library.utils import *
import torch 
from carla.self_explaining_model.catalog.vcnet.library.load_data import Load_dataset_carla
from torch import nn 
from torch.nn import functional as F
from carla.self_explaining_model.catalog.vcnet.library.vcnet_tabular_data_v1.split_immutable import *
from carla.data.catalog import DataCatalog

# Our join-training model
  
class CVAE_join_immutable(Load_dataset_carla) :

    '''
    Class for VCNet immutable version model architecture 
    ----------
    Parameters
    immutable_features : List 
    List of immutable features 
    cat_arrays : List 
    List of categorical features 
    cont_shape : Int 
    Number of continuous features 
    cat_arrays_mutable : List
    List of mutable categorical features 
    cont_shape_mutable : List 
    Number of mutable continuous features 
    ----- 
    Methods
    cat_normalize : function to obtain binary values from sigmoids for categorical features  
    encode_classif : pre-encoding for classif layers
    encode_generator : pre-encoding for cVAE 
    encode : cVAE encoding 
    reparameterize : reparametrization trick for cVAE
    decode : cVAE decoding 
    classif : classification encoding 
    forward : forward examples to the overall network 
    forward_counterfactuals : forward to the overall network and tweak the decoder to obtain counterfactuals 
    forward_pred : forward to obtain only predictions 
    '''

        
    def __init__(self, data_catalog : DataCatalog ,model_config_dict : Dict, immutable_features : list ,cat_arrays : list ,cont_shape : int,cat_arrays_mutable : list ,cont_shape_mutable : int):
        super().__init__(data_catalog,model_config_dict)
        # Dictionary that contains input parameters (in order to re-load the model for post-hoc comparison)
        self.kwargs = {"data_catalog" : data_catalog, "model_config_dict" : model_config_dict,"immutable_features" : immutable_features,"model_config_dict":model_config_dict,
                       "cat_arrays" : cat_arrays, "cont_shape" : cont_shape,  "cat_arrays_mutable" : cat_arrays_mutable,
                       "cont_shape_mutable" : cont_shape_mutable }
        
        
        self.cat_arrays = cat_arrays
        self.cont_shape = cont_shape
        self.cat_arrays_mutable = cat_arrays_mutable
        self.cont_shape_mutable = cont_shape_mutable
        self.immutable_features = immutable_features
        self.used_carla = True
        # Size of the input for the CVAE (change only mutable features)
        if len(cat_arrays_mutable) != 0 :
            self.feature_size_mutable = np.hstack(self.cat_arrays_mutable).flatten().shape[0] + self.cont_shape_mutable
        # if only numerical variables 
        else : 
            self.feature_size_mutable = self.cont_shape_mutable
        self.feature_size_immutable = self.feature_size - self.feature_size_mutable
        # Pre-encoding
        self.se1  = nn.Linear(self.feature_size, self.latent_size_share)
        
        self.se2 = nn.Linear(self.feature_size_mutable, self.latent_size_share)
        
        
        # C-VAE encoding 
        
        self.e1 = nn.Linear(self.latent_size_share,self.mid_reduce_size)
        self.e2 = nn.Linear(self.mid_reduce_size, self.latent_size)
        self.e3 = nn.Linear(self.mid_reduce_size, self.latent_size)
        
        # C-VAE Decoding

        self.fd1 = nn.Linear(self.latent_size + self.class_size-1 + self.feature_size_immutable, self.mid_reduce_size)
            
        self.fd2 = nn.Linear(self.mid_reduce_size, self.latent_size_share)
        self.fd3 = nn.Linear(self.latent_size_share, self.feature_size_mutable)
        
         # Classification 
        
        self.fcl1 = nn.Linear(self.latent_size_share,self.latent_size_share)
        self.fcl2 = nn.Linear(self.latent_size_share,self.class_size -1)
        # Activation functions
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
         
        
        
          

    # Softmax for counterfactual output + sigmoid on numerical variables (function in utils.py)
    def cat_normalize(self, c: torch.tensor, hard=False):
        return cat_normalize(c, self.cat_arrays_mutable, self.cont_shape_mutable,self.cont_shape_mutable,hard=hard,used_carla=self.used_carla)
        
    
    def encode_classif(self,x: torch.tensor) : 
        z = self.elu(self.se1(x))
        return(z)
    
    def encode_generator(self,x: torch.tensor) : 
        z = self.elu(self.se2(x))
        return(z)
    
    # C-VAE encoding 
    def encode(self, z: torch.tensor):
        inputs = z
        h1 = self.elu(self.e1(inputs))
        z_mu = self.e2(h1)
        z_var = self.e3(h1)
        return z_mu, z_var
    
    # Reparametrization trick 
    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor):
        #torch.manual_seed(0)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    # C-VAE decoding 
    def decode(self, z_prime: torch.tensor, c: torch.tensor): # P(x|z, c)
        inputs = torch.cat([z_prime, c], 1) # (bs, latent_size+class_size)
        h1 = self.elu(self.fd1(inputs))
        h2 = self.elu(self.fd2(h1)) 
        h3 = self.fd3(h2)
        return h3
    
    # Classification layers (after shared encoding)
    def classif(self,z: torch.tensor) :
        c1 = self.elu(self.fcl1(z))
        c2 = self.fcl2(c1)
        return self.sigmoid(c2)
     
    # Forward in train phase
    def forward(self, x: torch.tensor):
        x_imutable,x_mutable,_,_ = split_immutable(x, self.cat_arrays,self.continous_cols,self.discret_cols,self.immutable_features)
        
        z1 =  self.encode_classif(x)
        z2 = self.encode_generator(x_mutable)
        # Output of classification layers
        output_class = self.classif(z1)
        
        # C-VAE encoding  
        mu, logvar = self.encode(z2)
        z_prime = self.reparameterize(mu, logvar)
        
        # Decoded output 
        c = self.decode(z_prime, torch.hstack((output_class,x_imutable)))
        
         
        # Softmax activation for ohe variables 
        c = self.cat_normalize(c, hard=False)
       
        # Return Decoded output + output class
        return c, mu, logvar,output_class
    
    
    def forward_counterfactuals(self,x: torch.tensor,c_pred: torch.tensor) :
        
        
        x_imutable,x_mutable,_,_ = split_immutable(x, self.cat_arrays,self.continous_cols,self.discret_cols,self.immutable_features)
        
        z2 = self.encode_generator(x_mutable)
        
        
        mu, logvar = self.encode(z2)
        
        z_prime = self.reparameterize(mu, logvar)
        
        c_mutable = self.decode(z_prime, torch.hstack((c_pred,x_imutable)))
        
        # 0he format for c  
        c_mutable = self.cat_normalize(c_mutable, hard=True)
        
        # Return counterfactuals as concatenation of mutable and immutable features 
        continous_mutable = [e for e in self.continous_cols if e not in self.immutable_features]
        discrete_mutable = [e for e in self.discret_cols if e not in self.immutable_features]
        index_x =  cat_index_ohe(self.cat_arrays,self.continous_cols,self.discret_cols)
        index_x_mutable = cat_index_ohe(self.cat_arrays_mutable,continous_mutable,discrete_mutable)
        
        c = torch.clone(x)
        for e in index_x_mutable.keys() : 
            c[:,index_x[e][0]:index_x[e][1]] = c_mutable[:,index_x_mutable[e][0]:index_x_mutable[e][1]]
        
        return c, z_prime
    
    
    # Forward for prediction in test phase (prediction task)
    def forward_pred(self,x: torch.tensor) :
        z = self.encode_classif(x)
        # Output of classification layers
        output_class = self.classif(z)
        # Return classification layers output 
        return(output_class)
    
    
# Only predictor part of the network     
class Predictor(Load_dataset_carla) :

    '''
    Class for only the predictor part of VCNet immutable (useful to inclusion in Carla framework)
    '''

    def __init__(self, data_catalog : DataCatalog , model_config_dict : Dict, immutable_features : list ,cat_arrays : list ,cont_shape : int ,cat_arrays_mutable : list ,cont_shape_mutable : int):
        super().__init__(data_catalog,model_config_dict)
        
        # Dictionary that contains input parameters (in order to re-load the model for post-hoc comparison)
        self.kwargs = {"data_catalog" : data_catalog, "model_config_dict" : model_config_dict, "immutable_features" : immutable_features, "model_config_dict":model_config_dict,
                       "cat_arrays" : cat_arrays, "cont_shape" : cont_shape,  "cat_arrays_mutable" : cat_arrays_mutable,
                       "cont_shape_mutable" : cont_shape_mutable }
        
        self.cat_arrays = cat_arrays
        self.cont_shape = cont_shape
        self.cat_arrays_mutable = cat_arrays_mutable
        self.cont_shape_mutable = cont_shape_mutable
        self.immutable_features = immutable_features
        # Size of the input for the CVAE (change only mutable features)
        if len(cat_arrays_mutable) != 0 :
            self.feature_size_mutable = np.hstack(self.cat_arrays_mutable).flatten().shape[0] + self.cont_shape_mutable
        # if only numerical variables 
        else : 
            self.feature_size_mutable = self.cont_shape_mutable
        self.feature_size_immutable = self.feature_size - self.feature_size_mutable
        
        
        
        # Pre-encoding
        self.se1  = nn.Linear(self.feature_size, self.latent_size_share)
        
        self.se2 = nn.Linear(self.feature_size_mutable, self.latent_size_share)
        
        
        # Classification 
        
        self.fcl1 = nn.Linear(self.latent_size_share,self.latent_size_share)
        self.fcl2 = nn.Linear(self.latent_size_share,self.class_size -1)
        # Activation functions
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        
        
        
    def encode_classif(self,x : torch.tensor) : 
        z = self.elu(self.se1(x))
        return(z)
    
    # Classification layers (after shared encoding)
    def classif(self,z: torch.tensor) :
        c1 = self.elu(self.fcl1(z))
        c2 = self.fcl2(c1)
        return self.sigmoid(c2)
    
    
    # Forward for prediction in test phase (prediction task)
    def forward(self,x: torch.tensor) :
        z = self.encode_classif(x)
        # Output of classification layers
        output_class = self.classif(z)
        # Return classification layers output 
        return(output_class)
