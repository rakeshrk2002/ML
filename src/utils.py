import os
import sys 

import pandas as pd 
import numpy as np 
import dill as dill

from src.exception import Custom_Exception


def save_object(filepath, obj):
    
    try:
        
        dir_path = os.path.dirname(filepath)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(filepath,"wb") as file_obj:
            dill.dump(obj,filepath)
        
    except Exception as e:
        
        Custom_Exception(e, sys)
