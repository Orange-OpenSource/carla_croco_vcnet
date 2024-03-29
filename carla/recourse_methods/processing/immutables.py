# Software Name : carla_croco_vcnet
# Version: 1.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# the text of which is available at https://spdx.org/licenses/MIT.html
# or see the "license.txt" file for more details.

from typing import List


def encode_feature_names(imtbls: List[str], input_order: List[str]) -> List[str]:
    """
    Transforms not encoded immutable feature names into encoded ones

    Returns
    -------
    list
    """
    immutables = []
    # TODO: Maybe find a more elegant way to find encoded immutable feature names
    for feature in imtbls:
        if feature in input_order:
            immutables.append(feature)
        else:
            for cat in input_order:
                if cat not in immutables:
                    if feature in cat:
                        immutables.append(cat)
    return immutables
