import pandas as pd
from carla.evaluation.api import Evaluation
from carla.evaluation import remove_nans
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Uniform
import numpy as np 
from carla.models import MLModel
import torch.distributions.normal as normal_distribution
class Robustness(Evaluation):
    """
    Class to evaluate robustness with the recourse invalidation rate 
    -----
    Parameters : 
    -----
    Hyperparams : dict 
    Hyperparameters to compute robustness 
    mlmodel : carla.models.MLModel 
    Machine learning model to explain 
    -----
    Notes
    -----
    - Hyperparams

        * "n_samples": int
            Number of sample use for estimation 
        * "sigma2": float
            Variance of the noise distribution 
        * "distribution" : bool 
            Type of distribution, gaussian or uniform
    """
    def __init__(self, mlmodel : MLModel, hyperparameters : dict):
        super().__init__(mlmodel, hyperparameters)
        self.n_samples = self.hyperparameters["n_samples"]
        self.sigma2 = self.hyperparameters["sigma2"]
        self.distribution = self.hyperparameters["distribution"]
        self.mlmodel = mlmodel

    # Perturb samples with a gaussian distribution  
    def perturb_sample_gaussian(self,x : torch.tensor) :
        # stack copies of this sample, i.e. n rows of x.
        X = x.repeat(self.n_samples, 1)
        
        # sample normal distributed values
        Sigma = torch.eye(x.shape[1]) * self.sigma2
        eps = MultivariateNormal(
            loc=torch.zeros(x.shape[1]), covariance_matrix=Sigma
        ).sample((self.n_samples,))
        
        return X + eps, Sigma
    


    # Perturb samples with an uniform distribution 
    def perturb_sample_uniform(self,x : torch.tensor):
        # stack copies of this sample, i.e. n rows of x.
        X = x.repeat(self.n_samples, 1)
        
        # sample uniform distribution [x+sigma,x-sigma]
        eps = Uniform(-sigma2*torch.ones(x.shape[1]),self.sigma2*torch.ones(x.shape[1])).sample((self.n_samples,))
        
        
        return X + eps, None

    # Perturb samples with the choosen distribution
    def perturb_sample(self,x : torch.tensor ,distrib="gaussian"):
        if distrib=="gaussian" : 
            return(self.perturb_sample_gaussian(x))
        
        elif distrib=="uniform" : 
            return(self.perturb_sample_uniform(x))


    # Function to compute recourse invalidation rate 
    def _compute_robustness(self,counterfactuals: pd.DataFrame,backend="pytorch") : 
        result = []
        for i, x in counterfactuals.iterrows():
            x = torch.Tensor(x).unsqueeze(0)
            X_pert, _ = self.perturb_sample(x, distrib=self.distribution)
            if backend == "pytorch":
                delta_M = torch.mean(
                    (1 - (self.mlmodel.predict(X_pert).squeeze() > 0.5).int()).float()
                ).item()
            else:
                delta_M = np.mean(
                    1 - (self.mlmodel.predict(X_pert).squeeze() > 0.5).astype(int)
                )
            
            result.append(delta_M)
        
        results = pd.DataFrame(result,columns=["Robustness"])
        return(results)
    
    
    def get_evaluation(self, factuals: pd.DataFrame, counterfactuals: pd.DataFrame) :
        counterfactuals_without_nans = remove_nans(counterfactuals)
        # return empty dataframe if no successful counterfactuals
        if counterfactuals_without_nans.empty:
            return pd.DataFrame(columns=self.columns)
        else : 
            results = self._compute_robustness(counterfactuals_without_nans)
            return(results)