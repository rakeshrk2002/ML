import sys
import os
from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#class for data preparation for training the model
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):
      
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch", 
                "test preparation course",
            ]

            #Initiating pipeline for cleaning the data like handling missing values,one hot encoding etc

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Numerical columns: {numerical_columns}")

            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore',sparse=False)),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )
            logging.info("Transformation of the data has completed")

            return preprocessor
        
        except Exception as e:
            raise Custom_Exception(e,sys)
    
    # This function is for splitting cat and num data's for training and testing data and saving the preprocessor object for model training
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math score"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            print("input features : ")
            print(input_feature_train_df)
            print("---------------------")
            print("target feature : ")
            print(target_feature_train_df)
            print("---------------------")

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            print("Checking for missing values in categorical columns (Train Data):")
            print(input_feature_train_df.isnull().sum())

            print("Checking for missing values in categorical columns (Test Data):")
            print(input_feature_test_df.isnull().sum())

            for col in ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]:
                print(f"Unique values in train {col}: {input_feature_train_df[col].unique()}")
                print(f"Unique values in test {col}: {input_feature_test_df[col].unique()}")


            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise Custom_Exception(e,sys)