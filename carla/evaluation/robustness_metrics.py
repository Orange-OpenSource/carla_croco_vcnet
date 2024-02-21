
from carla.models.api import MLModel
import pandas as pd 
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Uniform
import numpy as np 
import torch.distributions.normal as normal_distribution
from torch.autograd import Variable

# Perturb samples with a normal distribution 
def perturb_sample(x, n_samples, sigma2,distrib="gaussian"):
    if distrib=="gaussian" : 
        return(perturb_sample_gaussian(x, n_samples, sigma2))
    
    elif distrib=="uniform" : 
        return(perturb_sample_uniform(x, n_samples, sigma2))

# Perturb samples with a gaussian distribution  
def perturb_sample_gaussian(x, n_samples, sigma2) :
    # stack copies of this sample, i.e. n rows of x.
    X = x.repeat(n_samples, 1)
    
    # sample normal distributed values
    Sigma = torch.eye(x.shape[1]) * sigma2
    eps = MultivariateNormal(
        loc=torch.zeros(x.shape[1]), covariance_matrix=Sigma
    ).sample((n_samples,))
    
    return X + eps, Sigma



# Perturb samples with an uniform distribution 
def perturb_sample_uniform(x, n_samples, sigma2):
    # stack copies of this sample, i.e. n rows of x.
    X = x.repeat(n_samples, 1)
    
    # sample uniform distribution [x+sigma,x-sigma]
    eps = Uniform(-sigma2*torch.ones(x.shape[1]),sigma2*torch.ones(x.shape[1])).sample((n_samples,))
    
    
    return X + eps, None


# Compute the recourse invalidation rate as a metric 
def compute_recourse_invalidation_rate(df_cfs: pd.DataFrame,model: MLModel, folder_name : str,n_samples,sigma2,backend="pytorch",distribution="gaussian") : 

    result = []
    cf_predictions = []
    
    for i, x in df_cfs.iterrows():
        x = torch.Tensor(x).unsqueeze(0)
        X_pert, _ = perturb_sample(x, n_samples, sigma2=sigma2,distrib=distribution)
        if backend == "pytorch":
            prediction = (model.predict(x).squeeze() > 0.5).int()
            cf_predictions.append(prediction.item())
            delta_M = torch.mean(
                (1 - (model.predict(X_pert).squeeze() > 0.5).int()).float()
            ).item()
        else:
            prediction = (model.predict(x).squeeze() > 0.5).astype(int)
            cf_predictions.append(prediction)
            delta_M = np.mean(
                1 - (model.predict(X_pert).squeeze() > 0.5).astype(int)
            )
        
        result.append(delta_M)
    df_cfs["prediction"] = cf_predictions
    
    
    results = pd.DataFrame(result)
    
    return(results)
    