from abc import abstractmethod
import pandas as pd
from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.data.catalog import DataCatalog
class SelfExplainingModel(MLModel,RecourseMethod):

    '''
    Class for self-explaining models, that makes prediction and explanation in the same network 
    It is a MLModel and RecourseMethod at the same time 

    Parameters
    ----------
    data : carla.data.Datacatalog
    Dataset to perform the method

    Methods
    -------
    get_counterfactuals : compute counterfactuals 
    '''

    def _init__(self,data:DataCatalog):
        super().__init__(data)

    @abstractmethod
    def get_counterfactuals(self, factuals: pd.DataFrame):
        """
        Generate counterfactual examples for given factuals.

        Parameters
        ----------
        factuals: pd.DataFrame
            Not encoded and not normalised factual examples in two-dimensional shape (m, n).

        Returns
        -------
        pd.DataFrame
            Encoded and normalised counterfactual examples.
        """
        pass