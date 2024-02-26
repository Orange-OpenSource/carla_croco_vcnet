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


class RecourseMethod(ABC):
    """
    Abstract class to implement custom recourse methods for a given black-box-model.

    Parameters
    ----------
    mlmodel: carla.models.MLModel
        Black-box-classifier we want to discover.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    encode_normalize_order_factuals:
        Uses encoder and scaler from black-box-model to preprocess data as needed.

    Returns
    -------
    None
    """

    def __init__(self, mlmodel):
        self._mlmodel = mlmodel

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
