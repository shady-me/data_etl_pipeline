import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import logging
import yaml
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataETLPipeline:
    """
    A comprehensive ETL pipeline for data preprocessing and transformation.
    Handles missing values, scaling, encoding, and feature engineering.
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config = self._load_config(config_path)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        
    def _load_config(self, config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to the configuration file.
            
        Returns:
            dict: Configuration dictionary.
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
            
    def extract(self, file_path=None):
        """
        Extract data from source.
        
        Args:
            file_path (str): Path to the data file. If None, uses path from config.
            
        Returns:
            pd.DataFrame: Loaded data.
        """
        try:
            path = file_path if file_path else self.config['data']['source_path']
            extension = os.path.splitext(path)[1].lower()
            
            if extension == '.csv':
                self.data = pd.read_csv(path)
            elif extension == '.json':
                self.data = pd.read_json(path)
            elif extension in ('.xls', '.xlsx'):
                self.data = pd.read_excel(path)
            elif extension == '.parquet':
                self.data = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
                
            logger.info(f"Data loaded successfully from {path}. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_data(self):
        """Validate the loaded data against configuration."""
        if self.data is None:
            raise ValueError("No data loaded. Call extract() first.")
            
        # Check required columns
        required_cols = self.config['data'].get('required_columns', [])
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
    def _build_preprocessor(self):
        """Build the preprocessing pipeline based on configuration."""
        numeric_features = self.config['preprocessing'].get('numeric_features', [])
        categorical_features = self.config['preprocessing'].get('categorical_features', [])
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.config['preprocessing'].get('numeric_impute_strategy', 'mean'))),
            ('scaler', StandardScaler() if self.config['preprocessing'].get('scaler') == 'standard' else MinMaxScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.config['preprocessing'].get('categorical_impute_strategy', 'most_frequent'))),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        logger.info("Preprocessor built successfully")
    
    def transform(self):
        """
        Apply all data transformations.
        
        Returns:
            tuple: Transformed training and test data (X_train, X_test, y_train, y_test)
        """
        self._validate_data()
        
        # Handle target variable
        target_col = self.config['data'].get('target_column')
        if target_col:
            y = self.data[target_col]
            
            # Encode if categorical
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                self.config['preprocessing']['label_encoder'] = le
                
            X = self.data.drop(columns=[target_col])
        else:
            X = self.data.copy()
            y = None
            
        # Split data if specified
        test_size = self.config['preprocessing'].get('test_size', 0.2)
        random_state = self.config['preprocessing'].get('random_state', 42)
        
        if y is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            self.X_train, self.X_test = train_test_split(
                X, test_size=test_size, random_state=random_state
            )
            self.y_train, self.y_test = None, None
            
        # Build and apply preprocessing
        self._build_preprocessor()
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        
        logger.info("Data transformation completed successfully")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def load(self, output_dir='output'):
        """
        Save processed data to disk.
        
        Args:
            output_dir (str): Directory to save processed data.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save processed data
            pd.DataFrame(self.X_train).to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
            pd.DataFrame(self.X_test).to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
            
            if self.y_train is not None:
                pd.DataFrame(self.y_train).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
                pd.DataFrame(self.y_test).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
            
            # Save preprocessing artifacts
            import joblib
            joblib.dump(self.preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
            
            logger.info(f"Processed data saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def run_pipeline(self, file_path=None):
        """
        Run the complete ETL pipeline.
        
        Args:
            file_path (str): Optional path to data file.
        """
        try:
            logger.info("Starting ETL pipeline")
            self.extract(file_path)
            self.transform()
            self.load(self.config['data'].get('output_dir', 'output'))
            logger.info("ETL pipeline completed successfully")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

# Example configuration (save as config.yaml)
"""
data:
  source_path: "data/raw_data.csv"
  output_dir: "data/processed"
  target_column: "target"
  required_columns:
    - "feature1"
    - "feature2"
    - "target"

preprocessing:
  numeric_features:
    - "feature1"
    - "feature2"
  categorical_features:
    - "category1"
    - "category2"
  numeric_impute_strategy: "median"
  categorical_impute_strategy: "most_frequent"
  scaler: "standard"
  test_size: 0.2
  random_state: 42
"""

if __name__ == "__main__":
    # Example usage
    pipeline = DataETLPipeline('config.yaml')
    pipeline.run_pipeline()