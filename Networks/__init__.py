#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .ERFNet import Net

# Add models if desired
model_dict = {'erfnet': Net}


def allowed_models():
    return model_dict.keys()


def define_model(mod, **kwargs):
    if mod not in allowed_models():
        raise KeyError("The requested model: {} is not implemented".format(mod))
    else:
        return model_dict[mod](**kwargs)
