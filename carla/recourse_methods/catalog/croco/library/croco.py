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
import numpy as np  
from typing import List, Optional
import torch 
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Uniform
import datetime 
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from carla.recourse_methods.processing import reconstruct_encoding_constraints

# Function to sample noises from gaussian or uniform distributions
def reparametrization_trick(mu,sigma2,device,n_samples,distrib) :
        
    if distrib=="gaussian" : 
        return(reparametrization_trick_gaussian(mu, sigma2, device,n_samples))
    
    elif distrib=="uniform" : 
        return(reparametrization_trick_uniform(mu, sigma2, device,n_samples))

 
# Gaussian 
def reparametrization_trick_gaussian(mu, sigma2, device,n_samples):
    
    #var = torch.eye(mu.shape[1]) * sigma2
    std = torch.sqrt(sigma2)
    epsilon = MultivariateNormal(loc=torch.zeros(mu.shape[1]), covariance_matrix=torch.eye(mu.shape[1]))
    epsilon = epsilon.sample((n_samples,))  # standard Gaussian random noise
    ones = torch.ones_like(epsilon)
    random_samples = mu.reshape(-1) * ones.to(device) + (std * epsilon).to(device)
    
    return random_samples


# Uniform 
def reparametrization_trick_uniform(x, sigma2, device,n_samples):
    
    epsilon = Uniform(torch.zeros(x.shape[1]),torch.ones(x.shape[1])).sample((n_samples,))
    ones = torch.ones_like(epsilon)
    random_samples = (x.reshape(-1) - sigma2*ones) + 2*sigma2*ones * epsilon
    
    return random_samples


''''
# Compute the invalidation rate from a torch model 
def compute_invalidation_rate(torch_model, random_samples):
    yhat = torch_model(random_samples.float())[:, 1]
    hat = (yhat > 0.5).float()
    ir = 1 - torch.mean(hat, 0)
    return ir
'''



# Solve the CROCO optimization problem (the goal is to find Perturb)
def Optimize_croco(model,x0,pred_class,delta,sigma2,n_samples,lr,max_iter,robustness_target,robustness_epsilon,cat_feature_indices,binary_cat_features,t,m,device,lambda_param,number_numerical,distribution) :
    # Target classes are 1, one hot encoded -> [0,1]
    y_target_class = torch.tensor([0,1]).float().to(device)
    y_target = y_target_class[1]
    G_target = torch.tensor(y_target).float().to(device)
    # Init lambda value 
    lamb = torch.tensor(lambda_param).float()
    # Init perturb 
    Perturb = Variable(torch.clone(delta.to(device)), requires_grad=True)
    

    x_cf_new = reconstruct_encoding_constraints(x0+Perturb,cat_feature_indices,binary_cat_features).to(device)
    
    # Set optimizer 
    optimizer = optim.Adam([Perturb], lr, amsgrad=True)
    
    # MSE loss for class term 
    loss_fn = torch.nn.MSELoss()
    
    # Prediction for the counterfactual 
    f_x_binary = model(x_cf_new.float())
    f_x = f_x_binary[:,1-pred_class]
    

    # Take random samples 
    random_samples = reparametrization_trick(x_cf_new, torch.tensor(sigma2), device, n_samples=n_samples,distrib=distribution)
    #invalidation_rate = compute_invalidation_rate(model, random_samples)
    
    G = random_samples
    
    # Translated group with reconstruct constraint such as categorical data is either 0 or 1 
    G_new = reconstruct_encoding_constraints(
        G, cat_feature_indices, binary_cat_features
    ).to(device)

    # Compute robustness constraint term 
    compute_robutness = (m + torch.mean(G_target- model((G_new).float())[:,1-pred_class])) / (1-t)
    
    
    #Lambda = []
    #Dist = []
    #Rob = []
    while (f_x <=t) and (compute_robutness > robustness_target + robustness_epsilon) : 
        it=0
        for it in range(max_iter) :
            
            optimizer.zero_grad()
            

            x_cf_new = reconstruct_encoding_constraints(x0+Perturb,cat_feature_indices,binary_cat_features)
            


            # Prediction for the counterfactual 
            f_x_binary = model(x_cf_new.float())
            f_x = f_x_binary[:,1-pred_class]
             
        
            
            
            
            # Take random samples 
            random_samples = reparametrization_trick(x_cf_new, torch.tensor(sigma2), device, n_samples=n_samples,distrib=distribution)
            #invalidation_rate = compute_invalidation_rate(model, random_samples)
            
            
            # New perturbated group translated 
            G = random_samples
            G_new = reconstruct_encoding_constraints(
                G, cat_feature_indices, binary_cat_features
            )
            
            # Compute (m + theta) / (1-t)
            mean_proba =  torch.mean(model((G_new).float())[:,pred_class])
            compute_robutness = (m + mean_proba) /(1-t)
            
            # Diff between robustness and targer robustness 
            robustness_invalidation = compute_robutness - robustness_target
            
            
            
            # Overall loss function 
            loss = robustness_invalidation**2 + loss_fn(f_x_binary,y_target_class) + lamb* torch.norm(Perturb,p=1)
            
            loss.backward()
            optimizer.step()
            
            
               
                
            
            it += 1
            
        #print("Theta",mean_proba)
        #print("compute_robutness",compute_robutness)
        #print("robustness_target",robustness_target)
        #print("lambda",lamb)
        #print("predicted_class",f_x)
        #Lambda.append(lamb.clone().detach().cpu().numpy())
        #Rob.append((robustness_invalidation**2).detach().cpu().numpy())
        #Dist.append(torch.norm(Perturb,p=1).detach().cpu().numpy())
        if (f_x > t) and ((compute_robutness < robustness_target + robustness_epsilon))  :
            print("Counterfactual Explanation Found")
            break
        
        
        lamb -= 0.25
        
        # Stop if no solution found for different lambda values 
        if lamb <=0 :
            print("No Counterfactual Explanation Found for these lambda values")
            break
        
    final_perturb = Perturb.clone()
    
    #np.savetxt("lambda_values_costERC",np.vstack(Lambda))
    #np.savetxt("rob_cost_ERC",np.vstack(Rob))
    #np.savetxt("dist_cost_ERC",np.vstack(Dist))
    return(final_perturb)




 

def croco(torch_model,
    x: np.ndarray,
    delta : np.ndarray,
    number_numerical : int,
    cat_feature_indices: List[int],
    binary_cat_features: bool = True,
    n_samples : int = 500,
    lr: float = 0.01,
    lambda_param: float = 1,
    sigma2 : float = 0.01,
    robustness_target : float = 0.3,
    robustness_epsilon : float = 0.01,
    n_iter: int = 1000,
    t : float = 0.5,
    m : float = 0.1,
    distribution : bool = "gaussian"
    ) -> np.ndarray:
    
    """ 
    The CROCO method 
    """
    device = "cpu"
    # Input example as a tensor 
    x0 = torch.from_numpy(x).float().to(device)
    # Target class
    pred_class = 0
    # Tensor init perturb
    delta = torch.from_numpy(delta)
    # Solve the the optimization problem 
    perturb = Optimize_croco(torch_model,x0,pred_class,delta,sigma2,n_samples,lr,n_iter,robustness_target,robustness_epsilon,cat_feature_indices,binary_cat_features,t,m,device,lambda_param,number_numerical,distribution=distribution)
    # New counterfactual
    x_new =(x0 + perturb).cpu().detach().numpy().squeeze(axis=0)
    return(x_new)









