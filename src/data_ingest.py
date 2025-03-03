from sklearn.model_selection import train_test_split
import pandas as pd
import os
import logging

log_dir="logs"
os.makedirs(log_dir, exist_ok=True)

logger=logging.getLogger('data_ingest')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler=logging.FileHandler(os.path.join(log_dir, 'data_ingest.log'))
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(config_path:str)->dict:
    pass

def read_data(data_path):
    """load data from csv file"""
    try:
        df=pd.read_csv(data_path)
        logger.debug("data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"error in loading data: {str(e)}")
        return None

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """preprocess data"""
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
        df.rename(columns={'v1':'label','v2':'text'}, inplace=True)

        logger.debug("data preprocessed successfully")
        return df
    except Exception as e:
        logger.error(f"error in preprocessing data: {str(e)}")
        return None

def save_data(train_df:pd.DataFrame, test_df:pd.DataFrame,data_dir:pd.DataFrame)->None:
    """save data to csv file"""
    try:
        raw_data=os.path.join(data_dir, 'raw')
        os.makedirs(raw_data, exist_ok=True)
        train_df.to_csv(os.path.join(raw_data, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(raw_data, 'test.csv'), index=False)
        logger.debug("data saved successfully")

    except Exception as e:
        logger.error(f"error in saving data: {str(e)}")

def main():

    try:     
        data_path="https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv"
        data_dir="./data"
        df=read_data(data_path)
        logger.debug("data loaded successfully")
        df=preprocess_data(df)
        logger.debug("data preprocessed successfully")
        train_df, test_df=train_test_split(df, test_size=0.2, random_state=42)
        logger.debug("data split successfully")
        save_data(train_df, test_df, data_dir)
    except Exception as e:
        logger.error(f"error in main: {str(e)}")

if __name__=="__main__":
    main()