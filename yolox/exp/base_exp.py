#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import ast
import pprint
from abc import ABCMeta, abstractmethod
from typing import Dict
from tabulate import tabulate

import torch
from torch.nn import Module

from yolox.utils import LRScheduler

#  ABCMeta是用于实现抽象类的一个基础类。我们抽象出一个基类，知道要有哪些方法，但只是抽象方法，并不实现功能，只能继承，而不能被实例化，但子类必须要实现该方法。
class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment."""

    def __init__(self):
        self.seed = None
        self.output_dir = "./YOLOX_outputs"
        self.print_interval = 100
        self.eval_interval = 10

    @abstractmethod  # 强制子类实现父类中指定的一个方法
    def get_model(self) -> Module:
        pass

    @abstractmethod
    def get_data_loader(
        self, batch_size: int, is_distributed: bool
    ) -> Dict[str, torch.utils.data.DataLoader]:
        pass

    @abstractmethod
    def get_optimizer(self, batch_size: int) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def get_lr_scheduler(
        self, lr: float, iters_per_epoch: int, **kwargs
    ) -> LRScheduler:
        pass

    @abstractmethod
    def get_evaluator(self):
        pass

    @abstractmethod
    def eval(self, model, evaluator, weights):
        pass

    def __repr__(self):
        table_header = ["keys", "values"]
        # vars() 函数返回对象object的属性和属性值的字典对象。
        # pprint.pformat() 函数会以字符串形式，返回列表或字典中的内容。
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        # This can be used for safely evaluating strings containing Python values from untrusted sources 
                        # without the need to parse the values oneself. It is not capable of evaluating arbitrarily complex expressions, 
                        # for example involving operators or indexing.
                        v = ast.literal_eval(v)
                setattr(self, k, v)
