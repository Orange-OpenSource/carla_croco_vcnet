

# Mandatory imports 
import matplotlib.pyplot as plt,numpy as np,pandas as pd,scipy
from typing import Union,Optional,Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import random

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate

from pathlib import Path

 
class NumpyDataset(TensorDataset):
    def __init__(self, *arrs, ):
        super(NumpyDataset, self).__init__()
        # init tensors
        # small patch: skip continous or discrete array without content
        self.tensors = [torch.tensor(arr).float() for arr in arrs if arr.shape[-1] != 0]
        assert all(self.tensors[0].size(0) == tensor.size(0) for tensor in self.tensors)

    def data_loader(self, batch_size=128, shuffle=True, num_workers=4):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def features(self, test=False):
        return tuple(self.tensors[:-1] if not test else self.tensors)

    def target(self, test=False):
        return self.tensors[-1] if not test else None

class PandasDataset(NumpyDataset):
    def __init__(self, df: pd.DataFrame):
        cols = df.columns
        X = df[cols[:-1]].to_numpy()
        y = df[cols[-1]].to_numpy()
        super().__init__(X, y)

# Fix all seed for the experiment 
def fix_seed(seed=42) : 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Load a json file 
def load_json(file_name: str):
    with open(file_name) as json_file:
        return json.load(json_file)

# Activation functions for categorical and numerical features 
def cat_normalize(c, cat_arrays, cat_idx,cont_shape,hard=False,used_carla=True):
    c_copy = c.clone()
    # categorical feature starting index
    for col in cat_arrays:
        
        cat_end_idx = cat_idx + len(col)
        if hard:
            if used_carla : 
                c_copy[:, cat_idx: cat_end_idx] = torch.round(F.sigmoid(c[:, cat_idx: cat_end_idx].clone())).long()
                
            else :
                c_copy[:, cat_idx: cat_end_idx] = F.gumbel_softmax(c[:, cat_idx: cat_end_idx].clone(), hard=hard)
        else:
            if used_carla : 
                c_copy[:, cat_idx: cat_end_idx] = F.sigmoid(c[:, cat_idx: cat_end_idx].clone())
            else : 
                # Softmax for categorical variables 
                c_copy[:, cat_idx: cat_end_idx] = F.softmax(c[:, cat_idx: cat_end_idx].clone(), dim=-1)
        cat_idx = cat_end_idx
    
        
    # Relu for all continious variables 
    c_copy[:,:cont_shape] = torch.sigmoid(c[:,:cont_shape].clone())
     
    return c_copy

# Round columns in list_int 
def int_round_dataset(dataset,dico_round) : 
    dataset = dataset.copy()
    list_round = list(dico_round.keys())
    for column in list_round : 
        dataset[column] = dataset[column].apply(lambda x : round(x,dico_round[column]))
    return dataset


# Check if inverse min max scaler and rounding give the same results as before 
def check_min_max_scaler(dataset,original_examples,dico_round) : 
    # Round numerical values 
    original_examples_round = int_round_dataset(original_examples,dico_round) 
    # Round the original dataset 
    original_true_round = dataset.data[dataset.data.columns[:-1]]
    original_true_round = original_true_round[original_examples_round.columns]
    original_true_round = int_round_dataset(original_true_round,dico_round) 
    # Check if obtain the same values 
    index = len(original_true_round) - len(original_examples_round)
    original_true_round = original_true_round.loc[index:]
    same = np.array_equal(original_true_round.values,original_examples_round.values)
    return(same,original_examples_round)
    

# Back to the transform space 
def back_to_min_max(original_counterfactuals,dataset) :
    X_cont = dataset.normalizer.transform(original_counterfactuals[dataset.continous_cols]) if dataset.continous_cols else np.array([[] for _ in range(len(original_counterfactuals))])
    X_cat = dataset.encoder.transform(original_counterfactuals[dataset.discret_cols]) if dataset.discret_cols else np.array([[] for _ in range(len(original_counterfactuals))])
    X = np.concatenate((X_cont, X_cat), axis=1)
    return(torch.from_numpy(X).float())

'''
from carla.self_explaining_model.catalog.vcnet.library.plot_distributions import numpy_to_dataframe
# Round counterfactuals numerical values for int 
def round_counterfactuals(X,results,dataset,training,dico_round) :
    # Examples and counterfactuals back to the original space 
    original_examples,original_counterfactuals = numpy_to_dataframe(X,results["cf"],dataset)
    orginal_examples_round = int_round_dataset(original_examples,dico_round) 
    #problem,orginal_examples_round = check_min_max_scaler(dataset,original_examples,dico_round)
    # Check if no problem with min max scaler of examples to explain 
    #print("Problem with rounding is",not problem)
    # Round counterfactuals columns as int 
    original_counterfactuals_round = int_round_dataset(original_counterfactuals,dico_round)
    # Counterfactuals in transform space with rounded values in original space 
    counterfactuals_min_max_round = back_to_min_max(original_counterfactuals_round,dataset)
    # Predicted proba for counterfactuals 
    proba_c = training.model.forward_pred(counterfactuals_min_max_round).squeeze(1)
    # percentage of counterfactuals that change predicted class when rounding 
    y_c = torch.round(proba_c).long()
    percentage = sum(y_c == results["y_c"])/results["y_c"].shape[0]
    # if not 100% validity
    if not bool(percentage==1.0) :
        print("Alert : Some class have changed after rounding")
        print("{:.2%} of counterfactuals are still valid".format(percentage))
        
    return {"X" : orginal_examples_round, # Examples
            "cf" : original_counterfactuals_round, # Counterfactuals
            "y_x" : results["y_x"], # Predicted examples classes
            "y_c" : y_c , #Predicted counterfactuals classes 
            "proba_x" : results["proba_x"], # Predicted probas for examples 
            "proba_c" : proba_c #Predicted probas for counterfactuals  
            }
'''
