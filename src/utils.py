import os
import sys 

import pandas as pd 
import numpy as np 
import dill as dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import Custom_Exception


def save_object(filepath, obj):

    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
        if os.path.getsize(filepath) > 0:
            print(f"Successfully saved {filepath} ({os.path.getsize(filepath)} bytes)")
        else:
            raise Custom_Exception(f"Failed to save {filepath} - empty file", sys)
            
    except Exception as e:
        raise Custom_Exception(e, sys)
        
def evaluate_model(X_train,y_train,X_test,y_test,models,params,n_jobs=3,cv=3,verbose=1,refit=False):
    try:
        
        report = {}
        
        for i in range(len(list(models))):

            model = list(models.values())[i]
            params = params[list(models.keys())[i]]
            
            gs = GridSearchCV(model,params,cv = cv,n_jobs=n_jobs,verbose=verbose,refit=refit)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report   
    
    except Exception as e :
        
        raise Custom_Exception(e,sys)
    
def load_object(file_path):
    try:
        print(f"Trying to load {file_path}")
        print(f"File size: {os.path.getsize(file_path)} bytes")
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise Custom_Exception(e, sys)
        
