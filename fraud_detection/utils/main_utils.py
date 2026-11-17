import os
import sys
import pandas as pd
import numpy as np 
import dill 
import yaml
from fraud_detection.exception.exception import FraudDetectionException
from fraud_detection.logging.logger import logging

def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e :
        raise FraudDetectionException(e, sys)

def write_yaml_file(file_path: str, content:dict, replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file)
    except Exception as e:
        raise FraudDetectionException(e, sys)

def save_object(file_path:str, obj:object)->None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise FraudDetectionException(e, sys)  

def load_object(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file :{file_path} does not exist")
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise FraudDetectionException(e, sys)

def save_numpy_array_data(file_path:str, array:np.array):
    try:
        logging.info('saving numpy array data')
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise FraudDetectionException(e, sys)

def load_numpy_array_data(file_path:str)->np.array:
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise FraudDetectionException(e, sys)