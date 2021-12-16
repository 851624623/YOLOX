#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import importlib
import os
import sys


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
    import yolox

    # yolox.__file__:/home/liuhongyu/Projects/YOLOX/yolox/__init__.py
    # 文件夹后面接.__file__，得到***/__init__.py；如果是py文件后面接.__file__，就得到该py文件的绝对路径
    yolox_path = os.path.dirname(os.path.dirname(yolox.__file__)) # 得到项目的绝对路径

    filedict = {
        "yolox-s": "yolox_s.py",
        "yolox-m": "yolox_m.py",
        "yolox-l": "yolox_l.py",
        "yolox-x": "yolox_x.py",
        "yolox-tiny": "yolox_tiny.py",
        "yolox-nano": "nano.py",
        "yolov3": "yolov3.py",
    }
    filename = filedict[exp_name]
    exp_path = os.path.join(yolox_path, "exps", "default", filename)
    return get_exp_by_file(exp_path)


def get_exp(exp_file, exp_name):
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
