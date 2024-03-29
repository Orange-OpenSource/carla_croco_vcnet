# Software Name : carla_croco_vcnet
# Version: 1.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# the text of which is available at https://spdx.org/licenses/MIT.html
# or see the "license.txt" file for more details.


from typing import Any

import numpy as np
import pandas as pd


def predict_negative_instances(model: Any, data: pd.DataFrame) -> pd.DataFrame:
    """Predicts the data target and retrieves the negative instances. (H^-)

    Assumption: Positive class label is at position 1

    Parameters
    ----------
    name : Tensorflow or PyTorch Model
        Model object retrieved by :func:`load_model`
    data : pd.DataFrame
        Dataset used for predictions
    Returns
    -------
    df :  data.api Data() class with negative predicted instances
    """
    # get processed data and remove target
    df = data.copy()
    df["y"] = predict_label(model, df)
    df = df[df["y"] == 0]
    df = df.drop("y", axis="columns")

    return df


def predict_label(model: Any, df: pd.DataFrame, as_prob: bool = False) -> np.ndarray:
    """Predicts the data target

    Assumption: Positive class label is at position 1

    Parameters
    ----------
    name : Tensorflow or PyTorch Model
        Model object retrieved by :func:`load_model`
    df : pd.DataFrame
        Dataset used for predictions
    Returns
    -------
    predictions :  2d numpy array with predictions
    """

    predictions = model.predict(df)

    if not as_prob:
        predictions = predictions.round()

    return predictions
