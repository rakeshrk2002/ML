import os
import sys
from src.exception import Custom_Exception
from src.logger import logging
import pandas as pd 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# from src.components.data_transformation import DataTransformation,DataTransformation_config

# Data_Ingestion_Config class is created to create train,test and data.csv files in the specified location ("artifacts" folder)
@dataclass
class Data_Ingestion_Config:
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    raw_data_path: str = os.path.join("artifacts","data.csv")
    
class Data_Ingestion:
    def __init__(self):
        self.ingestion_config = Data_Ingestion_Config()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        
        try:
            
            df = pd.read_csv(r"C:\Users\018017\OneDrive - Sify Technologies Limited\Documents\ineuron\ML_Project\src\notebook\data\Stud_performance.csv")
            logging.info("Read's the dataset as Dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            
            logging.info("Train Test split initiated")
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index = False,header =True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False,header =True)
            
            logging.info("Data Ingestion Completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path   
            )
  
        except Exception as e:
            raise Custom_Exception(e,sys)
        
        
if __name__=="__main__":
    obj = Data_Ingestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    from src.components.data_transformation import DataTransformation
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
    
