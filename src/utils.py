# import os
# import sys 

# import pandas as pd 
# import numpy as np 
# import dill as dill
# import pickle

# from sklearn.metrics import r2_score
# from sklearn.model_selection import GridSearchCV

# from src.exception import Custom_Exception
# from src.logger import logging

# def save_object(file_path: str, obj: object) -> None:
#     """
#     Save a Python object to a file using both pickle and joblib as fallback
#     """
#     try:
#         dir_path = os.path.dirname(file_path)
#         os.makedirs(dir_path, exist_ok=True)
        
#         # Try joblib first
#         try:
#             logging.info(f"Attempting to save with joblib to {file_path}")
#             joblib.dump(obj, file_path)
#             if os.path.getsize(file_path) > 0:
#                 logging.info(f"Successfully saved with joblib ({os.path.getsize(file_path)} bytes)")
#                 return
#         except Exception as e:
#             logging.warning(f"Joblib save failed: {str(e)}")
#             if os.path.exists(file_path):
#                 os.remove(file_path)
        
#         # Try pickle as fallback
#         logging.info(f"Attempting to save with pickle to {file_path}")
#         with open(file_path, 'wb') as f:
#             pickle.dump(obj, f)
        
#         if os.path.getsize(file_path) == 0:
#             raise ValueError("Saved file is empty")
            
#         logging.info(f"Successfully saved with pickle ({os.path.getsize(file_path)} bytes)")
        
#     except Exception as e:
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         logging.error(f"Error saving object: {str(e)}")
#         raise Custom_Exception(f"Failed to save object: {str(e)}", sys)

# def load_object(file_path: str) -> object:
#     """
#     Load a Python object from a file using both joblib and pickle
#     """
#     try:
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found: {file_path}")
            
#         if os.path.getsize(file_path) == 0:
#             raise ValueError(f"File is empty: {file_path}")
            
#         # Try joblib first
#         try:
#             return joblib.load(file_path)
#         except Exception:
#             # Try pickle as fallback
#             with open(file_path, 'rb') as f:
#                 return pickle.load(f)
                
#     except Exception as e:
#         raise Custom_Exception(f"Error loading object: {str(e)}", sys)



# def evaluate_model(X_train,y_train,X_test,y_test,models,params,n_jobs=3,cv=3,verbose=1,refit=False):
#     try:
        
#         report = {}
        
#         for i in range(len(list(models))):

#             model = list(models.values())[i]
#             params = params[list(models.keys())[i]]
            
#             gs = GridSearchCV(model,params,cv = cv,n_jobs=n_jobs,verbose=verbose,refit=refit)
#             gs.fit(X_train,y_train)
            
#             model.set_params(**gs.best_params_)
#             model.fit(X_train,y_train)
            
#             y_train_pred = model.predict(X_train)
#             y_test_pred = model.predict(X_test)
            
#             train_model_score = r2_score(y_train,y_train_pred)
#             test_model_score = r2_score(y_test,y_test_pred)
            
#             report[list(models.keys())[i]] = test_model_score
            
#         return report   
    
#     except Exception as e :
        
#         raise Custom_Exception(e,sys)

# # def save_object(file_path: str, obj: object) -> None:
# #     try:
# #         logging.info(f"Starting to save object to {file_path}")
        
# #         # Create directory if it doesn't exist
# #         dir_path = os.path.dirname(file_path)
# #         os.makedirs(dir_path, exist_ok=True)
        
# #         # Verify object
# #         if obj is None:
# #             raise ValueError("Cannot save None object")
            
# #         # Save directly to final location (previous temp file approach may be causing issues)
# #         with open(file_path, "wb") as file_obj:
# #             pickle.dump(obj, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
# #             file_obj.flush()  # Ensure all data is written
# #             os.fsync(file_obj.fileno())  # Force write to disk
            
# #         # Verify the saved file
# #         if not os.path.exists(file_path):
# #             raise FileNotFoundError(f"Failed to create file: {file_path}")
            
# #         file_size = os.path.getsize(file_path)
# #         if file_size == 0:
# #             raise ValueError(f"Saved file is empty: {file_path}")
            
# #         # Verify we can load the object back
# #         with open(file_path, 'rb') as file_obj:
# #             loaded_obj = pickle.load(file_obj)
# #             if loaded_obj is None:
# #                 raise ValueError("Loaded object is None")
                
# #         logging.info(f"Successfully saved object to {file_path}")
# #         logging.info(f"File size: {file_size} bytes")
        
# #     except Exception as e:
# #         logging.error(f"Error in save_object: {str(e)}")
# #         # If file exists but is empty/corrupt, remove it
# #         if os.path.exists(file_path):
# #             os.remove(file_path)
# #         raise Custom_Exception(e, sys)



    
# # def load_object(file_path: str) -> object:
# #     try:
# #         if not os.path.exists(file_path):
# #             raise FileNotFoundError(f"File not found: {file_path}")
            
# #         if os.path.getsize(file_path) == 0:
# #             raise ValueError(f"File is empty: {file_path}")
            
# #         with open(file_path, 'rb') as file_obj:
# #             obj = pickle.load(file_obj)
            
# #         if obj is None:
# #             raise ValueError("Loaded object is None")
            
# #         return obj
        
# #     except Exception as e:
# #         raise Custom_Exception(f"Error in load_object: {str(e)}", sys)
import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import Custom_Exception

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise Custom_Exception(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train) # Train model

            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise Custom_Exception(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise Custom_Exception(e, sys)