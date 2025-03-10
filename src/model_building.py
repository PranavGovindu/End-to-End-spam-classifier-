import os
import numpy as np
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    try:
        
        
        logger.debug('Initializing RandomForest model with parameters: %s', params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        
        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        
        return clf
    
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise




def save_model_onnx(model, file_path: str, X_train: np.ndarray) -> None:
    """
    Save the trained model to ONNX format.

    :param model: Trained model object
    :param file_path: Path to save the ONNX model
    :param X_train: Training features (used for defining the input shape)
    """
    try:
        # Define the input type
        initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

        # Convert the model to ONNX format
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the ONNX model
        with open(file_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())

        logger.debug('Model saved to ONNX format at %s', file_path)

    except Exception as e:
        logger.error('Error occurred while saving the model to ONNX: %s', e)
        raise


def main():
    try:
        params = load_params('./params.yaml')['model_building']

        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)
        
       
        onnx_save_path = 'models/model.onnx'
        save_model_onnx(clf, onnx_save_path, X_train)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()