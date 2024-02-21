################################################################
# Software Name : CROCO
# Version: 0.1
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 Orange
#
# This software is confidential and proprietary information of Orange.
# You shall not disclose such Confidential Information and shall not copy, use or distribute it
# in whole or in part without the prior written consent of Orange
#
# Author: Guyomard Victor
##################################################################################



from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.croco.library import croco
import pandas as pd 
from carla.models import MLModel
from carla.recourse_methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
)
import numpy as np

class CROCO(RecourseMethod) : 
    """
    This class implemented counterfactual generation for the CROCO method 

    Parameters
    ----------
    mlmodel : carla.models.MLModel
        Machine learning model to explain 
    hyperparams : dict
        Dictionary containing hyperparameters. See Notes below to see its content.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "n_samples": int
            Number of sample to estimate the recourse invalidation rate estimator 
        * "lr": float
            Learning rate for gradient descent 
        * "t": float
            Decision threeshold for the machine learning model 
        * "m": float
            Hyperparameter for the bound tightness 
        * "lambda_param" : float
            Hyperparameter to quantify the distance/recourse trade-off (automatically adjust during training)
        * "n_iter" : int
            Number of grandient steps
        * "binary_cat_features" : bool 
            Set to true if the categorical features are encoded with "OneHot_drop_binary"
        * "sigma2" : float
            Variance for the noise 
        * "robustness_target" : float
            Target for the upper-bound 
        * "robustness_epsilon" :   float
            Tolerance to check robustness constraint 
        * "distribution" : str 
            Distribution for noise (gaussian or uniform)


    """
    
    _DEFAULT_HYPERPARAMS = {
        "n_samples" : 500,
        "lr": 0.01,
        "t" : 0.5, 
        "m" : 0.1,
        "lambda_param": 1,
        "n_iter": 1000,
        "binary_cat_features": True,
        "sigma2" : 0.01, 
        "robustness_target" : 0.3,
        "robustness_epsilon" : 0.01,
        "distribution" : "gaussian"
    }
    
    
    def __init__(self, mlmodel : MLModel , hyperparams : dict):
        super().__init__(mlmodel)
    
        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        self.n_samples = checked_hyperparams["n_samples"]
        self._lr = checked_hyperparams["lr"]
        self._lambda_param = checked_hyperparams["lambda_param"]
        self._n_iter = checked_hyperparams["n_iter"]
        self._binary_cat_features = checked_hyperparams["binary_cat_features"]
        self._sigma2 =  checked_hyperparams["sigma2"]
        self.robustness_target = checked_hyperparams["robustness_target"]
        self.robustness_epsilon = checked_hyperparams["robustness_epsilon"]
        self.distribution = checked_hyperparams["distribution"]
        self.m = checked_hyperparams["m"]
        self.t = checked_hyperparams["t"]
        
    def get_counterfactuals(self,factuals: pd.DataFrame) -> pd.DataFrame :
            
        encoded_feature_names = self._mlmodel.data.encoder.get_feature_names(
            self._mlmodel.data.categorical
        )
        
        # Index of categorical features 
        cat_features_indices = [
            factuals.columns.get_loc(feature)
            for feature in encoded_feature_names
        ]
        
        
        # Number of numerical features 
        number_numerical = len(self._mlmodel.data.continuous)
        
        
        # Compute robust counterfactuals for every x instance
        df_cfs_new = factuals.copy()
        for index, x in factuals.iterrows() :
                # Init perturb as zeros 
                perturb_init = np.zeros(x.shape)
                
                df_cfs_new.loc[index] = croco(self._mlmodel.raw_model,
                                                      np.array(x).reshape((1, -1)),
                                                      perturb_init,
                                                      number_numerical,
                                                      cat_features_indices,
                                                      binary_cat_features=self._binary_cat_features,
                                                      n_samples = self.n_samples,
                                                      lr=self._lr,
                                                      lambda_param=self._lambda_param,
                                                      sigma2 = self._sigma2,
                                                      robustness_target = self.robustness_target,
                                                      robustness_epsilon = self.robustness_epsilon,
                                                      n_iter=self._n_iter,
                                                      t = self.t,
                                                      m = self.m, 
                                                      distribution = self.distribution
                                                      )
            
        
        
        df_cfs_new = check_counterfactuals(self._mlmodel, df_cfs_new, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs_new)
        return(df_cfs)
            
            
            
            
            
 
