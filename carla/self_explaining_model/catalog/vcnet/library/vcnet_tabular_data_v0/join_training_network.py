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
from carla.data.catalog import DataCatalog


# Our join-training model   
class CVAE_join(Load_dataset_carla) :

    '''
    Class for VCNet base version model architecture 
    ----------
    Parameters
    ablation : str
    Parameter for ablation study 
    condition : str 
    Parameter for ablation study 
    shared_layers : bool 
    Parameter for ablation study 
    debug_enc : bool 
    Parameter for encoder debugging 
    ----- 
    Methods
    cat_normalize : function to obtain binary values from sigmoids for categorical features  
    encode_share : pre-encoding share between predictor and cVAE
    encode_classif : pre-encoding only for classif layers (ablation study)
    encode_generator : pre-encoding for only cVAE (ablation study) 
    encode : cVAE encoding 
    reparameterize : reparametrization trick for cVAE
    decode : cVAE decoding 
    classif : classification encoding 
    forward : forward examples to the overall network 
    forward_counterfactuals : forward to the overall network and tweak the decoder to obtain counterfactuals 
    forward_pred : forward to obtain only predictions 
    '''

    def __init__(self, data_catalog : DataCatalog ,model_config_dict : dict, cat_arrays : list ,cont_shape : int ,ablation : str ,condition : str ,shared_layers=True,debug_enc=False):
        super().__init__(data_catalog,model_config_dict)
        
        # Dictionary that contains input parameters
        self.kwargs = {"data_catalog" : data_catalog, "model_config_dict" : model_config_dict,
                       "cat_arrays" : cat_arrays, "cont_shape" : cont_shape,  "ablation" : ablation,
                       "condition" : condition, "shared_layers" : shared_layers}
        
        self.cat_arrays = cat_arrays
        self.used_carla = True
        self.cont_shape = cont_shape
        self.ablation = ablation
        self.condition = condition 
        self.shared_layers = shared_layers
        self.debug_enc = debug_enc
        # Shared encoding
        self.se1  = nn.Linear(self.feature_size, self.latent_size_share)
        
        self.se2 = nn.Linear(self.feature_size, self.latent_size_share)
        
        
        # C-VAE encoding 
        # Remove condition on enc for ablation study
        if self.ablation == "remove_enc" : 
            self.e1 = nn.Linear(self.latent_size_share,self.mid_reduce_size)
        else  :
            self.e1  = nn.Linear(self.latent_size_share + self.class_size-1,self.mid_reduce_size)
        self.e2 = nn.Linear(self.mid_reduce_size, self.latent_size)
        self.e3 = nn.Linear(self.mid_reduce_size, self.latent_size)
        
        # C-VAE Decoding
        
        # Remove condition on dec for ablation study
        if self.ablation == "remove_dec"  :
            self.fd1 = nn.Linear(self.latent_size, self.mid_reduce_size)
        else : 
            self.fd1 = nn.Linear(self.latent_size + self.class_size-1, self.mid_reduce_size)
            
        self.fd2 = nn.Linear(self.mid_reduce_size, self.latent_size_share)
        self.fd3 = nn.Linear(self.latent_size_share, self.feature_size)
        
         # Classification 
        
        self.fcl1 = nn.Linear(self.latent_size_share,self.latent_size_share)
        self.fcl2 = nn.Linear(self.latent_size_share,self.class_size -1)
        # Activation functions
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        
        
         
          
    # Softmax for counterfactual output + sigmoid on numerical variables (function in utils.py)
    def cat_normalize(self, c: torch.tensor, hard=False):
        # categorical feature starting index
        cat_idx = len(self.continous_cols)
        return cat_normalize(c, self.cat_arrays, cat_idx,self.cont_shape,hard=hard,used_carla=self.used_carla)
        
    # Shared encoding     
    def encode_share(self,x: torch.tensor) :
        z = self.elu(self.se1(x))
        return(z)
    
    def encode_classif(self,x: torch.tensor) : 
        z = self.elu(self.se1(x))
        return(z)
    
    def encode_generator(self,x: torch.tensor) : 
        z = self.elu(self.se2(x))
        return(z)
    
    # C-VAE encoding 
    def encode(self, z: torch.tensor,c: torch.tensor):
        if self.ablation == "remove_enc" : 
            inputs = z
        else :
            inputs = torch.cat([z, c], 1) 
        h1 = self.elu(self.e1(inputs))
        z_mu = self.e2(h1)
        z_var = self.e3(h1)
        return z_mu, z_var
    
    # Reparametrization trick
    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std,)
        #print("mu",mu)
        #print("eps",eps)
        #print("std",std)
        return mu + eps*std
    
    # C-VAE decoding 
    def decode(self, z_prime: torch.tensor, c: torch.tensor): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        # Remove condition for ablation study 
        if self.ablation == "remove_dec" : 
            inputs = z_prime
        else : 
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
    def forward(self, x: torch.tensor,ohe=False):
        # if layers not shared between predictor and generator 
        if not self.shared_layers : 
            z1 =  self.encode_classif(x)
            z2 = self.encode_generator(x)
        else : 
            z1 = z2 =  self.encode_share(x)
        
        # Output of classification layers
        output_class = self.classif(z1)
        
        # C-VAE encoding  
        if self.debug_enc :
            mu = torch.zeros(x.shape[0],self.latent_size)
            logvar = torch.zeros(x.shape[0],self.latent_size) + 1 
        else : 
            mu, logvar = self.encode(z2,output_class)
        
        z_prime = self.reparameterize(mu, logvar)
        
        # Decoded output 
        c = self.decode(z_prime,output_class)
        
         
        # Softmax activation for ohe variables 
        c = self.cat_normalize(c, hard=ohe)
        #print("c",c)
        # Return Decoded output + output class
        return c, mu, logvar,output_class
    
    
    def forward_counterfactuals(self,x : torch.tensor,c_pred: torch.tensor) :
        
        # if layers not shared between predictor and generator 
        if not self.shared_layers : 
            z1 =  self.encode_classif(x)
            z2 = self.encode_generator(x)
        else : 
            z1 = z2 =  self.encode_share(x)
            
            # Output of classification layers
        output_class = self.classif(z1)
        
        # C-VAE encoding  
        if self.condition == "change_dec_only" :
            mu, logvar = self.encode(z2,output_class)
        elif self.condition == "change_enc_only" or self.condition == "change_enc_dec" : 
            
            mu, logvar = self.encode(z2,c_pred)
            
        if self.debug_enc :
            mu = torch.zeros(x.shape[0],self.latent_size)
            logvar = torch.zeros(x.shape[0],self.latent_size) + 1     
        
        z_prime = self.reparameterize(mu, logvar)
        
        
        # Decoded output 
        if self.condition == "change_dec_only" or self.condition == "change_enc_dec" :
            
            c = self.decode(z_prime, c_pred)
        elif self.condition == "change_enc_only" : 
            
            c = self.decode(z_prime, output_class)
        
        # 0he format for c 
        c = self.cat_normalize(c, hard=True)
        
        return c, z_prime
        
    # Forward for prediction in test phase (prediction task)
    def forward_pred(self,x: torch.tensor) :
        if not self.shared_layers : 
            z = self.encode_classif(x)
        else : 
            # Shared encoding 
            z = self.encode_share(x)
        
        # Output of classification layers
        output_class = self.classif(z)
            
        # Return classification layers output 
        return(output_class)
    
# Only predictor part of the network     
class Predictor(Load_dataset_carla) :
        
    '''
    Class for only the predictor part of VCNet (useful to inclusion in Carla framework)
    '''


    def __init__(self, data_catalog: DataCatalog , model_config_dict : Dict, cat_arrays : list ,cont_shape : int ,ablation : str ,condition : str ,shared_layers=True):
        super().__init__(data_catalog,model_config_dict)
        
        # Dictionary that contains input parameters 
        self.kwargs = {"data_catalog" : data_catalog, "model_config_dict" : model_config_dict,
                       "cat_arrays" : cat_arrays, "cont_shape" : cont_shape,  "ablation" : ablation,
                       "condition" : condition, "shared_layers" : shared_layers}
        
        self.cat_arrays = cat_arrays
        self.cont_shape = cont_shape
        self.ablation = ablation
        self.condition = condition 
        self.shared_layers = shared_layers
        
        
        # Shared encoding
        self.se1  = nn.Linear(self.feature_size, self.latent_size_share)
        
        self.se2 = nn.Linear(self.feature_size, self.latent_size_share)
        
        
        # Classification 
       
        self.fcl1 = nn.Linear(self.latent_size_share,self.latent_size_share)
        self.fcl2 = nn.Linear(self.latent_size_share,self.class_size -1)
        # Activation functions
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        
        
        
    # Shared encoding     
    def encode_share(self,x : torch.tensor ) :
        z = self.elu(self.se1(x))
        return(z)
    
    def encode_classif(self,x: torch.tensor) : 
        z = self.elu(self.se1(x))
        return(z)
    
    
    # Classification layers (after shared encoding)
    def classif(self,z: torch.tensor) :
        c1 = self.elu(self.fcl1(z))
        c2 = self.fcl2(c1)
        return self.sigmoid(c2)
    
    
    # Forward for prediction in test phase (prediction task)
    def forward(self,x: torch.tensor) :
        if not self.shared_layers : 
            z = self.encode_classif(x)
        else : 
            # Shared encoding 
            z = self.encode_share(x)
        
        # Output of classification layers
        output_class = self.classif(z)
        
        # Return classification layers output 
        return(output_class)