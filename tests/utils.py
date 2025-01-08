'''Utility functions for testing suite'''
import yaml
import os
def read_yaml(path):
    with open(path,'rb') as stream:
        out = yaml.safe_load(stream)
    return out