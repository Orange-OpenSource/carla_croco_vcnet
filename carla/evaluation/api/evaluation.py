# Software Name : carla_croco_vcnet
# Version: 1.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# the text of which is available at https://spdx.org/licenses/MIT.html
# or see the "license.txt" file for more details.


from abc import ABC, abstractmethod

import pandas as pd


class Evaluation(ABC):
    def __init__(self, mlmodel, hyperparameters: dict = None):
        """

        Parameters
        ----------
        mlmodel:
            Classification model. (optional)
        hyperparameters:
            Dictionary with hyperparameters, could be used to pass other things. (optional)
        """
        self.mlmodel = mlmodel
        self.hyperparameters = hyperparameters

    @abstractmethod
    def get_evaluation(
        self, factuals: pd.DataFrame, counterfactuals: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute evaluation measure"""
