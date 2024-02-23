# Software Name : carla_croco_vcnet
# Version: 1.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# the text of which is available at https://spdx.org/licenses/MIT.html
# or see the "license.txt" file for more details.

from typing import Dict
from typing import Union

import torch 

import numpy as np
import pandas as pd
from carla.data.catalog import DataCatalog
from carla.self_explaining_model.api import SelfExplainingModel
from carla.recourse_methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
)
from carla.self_explaining_model.catalog.vcnet.library.vcnet_tabular_data_v0.join_training_network import CVAE_join
from carla.self_explaining_model.catalog.vcnet.library.vcnet_tabular_data_v1.join_training_network import CVAE_join_immutable
from carla.self_explaining_model.catalog.vcnet.library.vcnet_tabular_data_v1.split_immutable import *

from carla.self_explaining_model.catalog.vcnet.library.train_network import Train_CVAE_base,Train_CVAE_immutable,Init_training
from carla.self_explaining_model.catalog.vcnet.library.load_data import Load_dataset_carla
from carla.self_explaining_model.catalog.vcnet.library.load_classif_model import load_classif_model

class VCNet(SelfExplainingModel) :
    """
    Implementation of VCNet [1]_

    Parameters
    ----------
    data : carla.data.Datacatalog
        Dataset to perform the method
    hyperparams : dict
        Dictionary containing hyperparameters. See Notes below to see its content.

    Methods
    -------
    predict:
        One-dimensional prediction of ml model for an output interval of [0, 1].
    predict_proba:
        Two-dimensional probability prediction of ml model.
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "data_name": str
            name of the dataset
        * "vcnet_params": Dict
            Hyperparameters for VCnet training 

            + "train": bool
                Decides if VCnet is trained.
            + "lr": float, default: 1e-3
                Learning rate for VCnet
            + "batch_size": int
                Batch-size for VCnet
            + "epochs": int
                Number of epochs for training
            + "lambda_1": float
                Loss function weight 
            + "lambda_2": float
                Loss function weight 
            + "lambda_3": float
                Loss function weight 
            + "latent_size": float
                Loss function weight 
            + "latent_size_share": float
                Loss function weight 
            + "mid_reduce_size": float
                Loss function weight 
            + "kld_start_epoch" : float
                Update hyperparameter KLD
            + "max_lambda_1 : float 
                Max value for KLD loss weight 
            

    .. [1] Guyomard
    """

    _DEFAULT_HYPERPARAMS = {   
    "name" : None ,
    "vcnet_params" : {
    "train" : False,
    "lr":  1.14e-5,
    "batch_size": 91,
    "epochs" : 176,
    "lambda_1": 0,
    "lambda_2": 0.93,
    "lambda_3": 1,
    "latent_size" : 19,
    "latent_size_share" :  304, 
    "mid_reduce_size" : 152,
    "kld_start_epoch" : 0.112,
    "max_lambda_1" : 0.034 }
    }
    

    def __init__(self,data : DataCatalog ,hyperparams : dict, immutable_features : list, train_cvae=False,immutable=False) :
        super().__init__(data)
        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        make_train = checked_hyperparams["vcnet_params"]["train"]
        del checked_hyperparams["vcnet_params"]["train"]
        checked_hyperparams = {"name" : checked_hyperparams["name"],**checked_hyperparams["vcnet_params"]}
        
        # Create a load dataset object

        self.dataset = Load_dataset_carla(data,checked_hyperparams)
        self.immutable_features = immutable_features
        self.immutable = immutable
        # Prepare dataset and return dataloaders + categorical index 
        loaders,cat_arrays,cont_shape = self.dataset.prepare_data()
        
        if self.immutable : 
            # Select parameters for mutable variables 
            _,_,cont_shape_mutable,cat_arrays_mutable = split_immutable(self.dataset.train_dataset[:][0], cat_arrays,self.dataset.continous_cols,self.dataset.discret_cols,self.immutable_features)
            ### Prepare training 
            self._training = Train_CVAE_immutable(data,checked_hyperparams,cat_arrays,cont_shape,loaders,self.immutable_features,cat_arrays_mutable,cont_shape_mutable,cuda_name="cpu")
        else : 
            ### Prepare training 
            self._training = Train_CVAE_base(data,checked_hyperparams,cat_arrays,cont_shape,loaders,ablation="remove_enc",condition="change_dec_only",cuda_name="cpu",shared_layers=True)

        if make_train :
            ### Train VCnet model
            self._training.train_and_valid_cvae(tensorboard=False)

        # Load Vcnet model weights  
        self._training.load_weights()

        # Extract predictor part of VCnet as a pytorch model
        self._model = load_classif_model(self._training,immutable=self.immutable)
           
    @property
    def feature_input_order(self):
        # this property contains a list of the correct input order of features for the ml model
        test = self.data.df_test.drop(columns=[self.dataset.target])
        self._feature_input_order = list(test)
        return self._feature_input_order

    @property
    def backend(self):
        """
        Describes the type of backend which is used

        E.g., tensorflow, pytorch, sklearn, xgboost

        Returns
        -------
        str
        """
        return "pytorch"
    
    @property
    def raw_model(self):
        """
        Contains the raw VCnet predictor model built on its framework

        Returns
        -------
        object
            Classifier, depending on used framework
        """
        return self._model

    def predict(self, x: Union[np.ndarray, pd.DataFrame]):
        # the predict function outputs the continuous prediction of the model, similar to sklearn.
        return torch.argmax(self._model.forward(torch.from_numpy(x.to_numpy()).float()),axis=1).detach().numpy()
        
    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]):
        
        x = self.get_ordered_features(x)

        if len(x.shape) != 2:
            raise ValueError("Input shape has to be two-dimensional")

        # Keep model and input on the same device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(device)

        if isinstance(x, pd.DataFrame):
            _x = x.values
        elif isinstance(x, torch.Tensor):
            _x = x.clone()
        else:
            _x = x.copy()

        # If the input was a tensor, return a tensor. Else return a np array.
        tensor_output = torch.is_tensor(x)
        if not tensor_output:
            _x = torch.Tensor(_x)

        # input, tensor_output = (
        #     (torch.Tensor(x), False) if not torch.is_tensor(x) else (x, True)
        # )

        _x = _x.to(device)
        output = self._model.forward(_x)

        if tensor_output:
            return output
        else:
            return output.detach().cpu().numpy()

    ### Function to generate counterfactuals 
    def get_counterfactuals(self , factuals : pd.DataFrame,eps=0.01,round=False) :
        data = torch.from_numpy(factuals.to_numpy()).float()
        results = self._training.compute_counterfactuals(data)
        
        # Round numerical values, x< eps implies no change
        if round : 
            results,_ = self._training.round_counterfactuals(results,eps,data)
            counterfactuals = results["cf"]
        else : 
            counterfactuals = results["cf"]
        
        

        df_cfs =  pd.DataFrame(counterfactuals.numpy(),columns=self.feature_input_order)
        # Replace unvalid counterfactuals by Nan values for metric evaluation 
        df_cfs = check_counterfactuals(self, df_cfs, factuals.index)
        df_cfs = self.get_ordered_features(df_cfs)
        return df_cfs
