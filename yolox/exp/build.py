#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import importlib
import os
import sys

# yolox.__file__:/home/liuhongyu/Projects/YOLOX/yolox/__init__.py
# 文件夹后面接.__file__，得到***/__init__.py；如果是py文件后面接.__file__，就得到该py文件的绝对路径
# yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))

def get_exp_by_file(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))
        # importlib.import_module(name=可以是字符串)，动态导入
        # 因为上面有sys.path.append，所以这步直接导入py名就行
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


def get_exp_by_name(exp_name):
    exp = exp_name.replace("-", "_")  # convert string like "yolox-s" to "yolox_s"
    module_name = ".".join(["yolox", "exp", "default", exp])
    exp_object = importlib.import_module(module_name).Exp()
    return exp_object


def get_exp(exp_file=None, exp_name=None):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file)
    else:
        return get_exp_by_name(exp_name)
