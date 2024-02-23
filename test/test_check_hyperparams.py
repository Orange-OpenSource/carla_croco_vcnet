# Software Name : carla_croco_vcnet
# Version: 1.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# the text of which is available at https://spdx.org/licenses/MIT.html
# or see the "license.txt" file for more details.

from carla.recourse_methods.processing.counterfactuals import merge_default_parameters


def test_check_hyperparams():
    hyperparams = {"key1": 1, "key3": "3", "key5": {"sub_key1": 22}}

    default = {
        "key1": 22,
        "key2": "_optional_",
        "key3": "1",
        "key4": {"sub_key1": 1, "sub_key2": 2},
        "key5": {"sub_key1": None},
    }

    actual = merge_default_parameters(hyperparams, default)

    expected = {
        "key1": 1,
        "key2": None,
        "key3": "3",
        "key4": {"sub_key1": 1, "sub_key2": 2},
        "key5": {"sub_key1": 22},
    }

    assert actual == expected
