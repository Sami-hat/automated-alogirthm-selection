import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


class DataHandler:
    """Handles data loading, preprocessing, and validation."""
    
    SUPPORTED_FORMATS = ['.txt', '.csv', '.npy', '.npz']
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self._validate_data_directory()
        self.scaler = None
        self.feature_names = None
        self.algorithm_names = None
        
    def _validate_data_directory(self):
        """Validate that the data directory exists and has required structure."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        required_subdirs = ['train', 'test']
        for subdir in required_subdirs:
            if not (self.data_dir / subdir).exists():
                raise FileNotFoundError(f"Required subdirectory '{subdir}' not found in {self.data_dir}")
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load training and test data with enhanced error handling."""
        try:
            # Load performance data
            train_perf = self._load_file(self.data_dir / 'train' / 'performance-data.txt')
            test_perf = self._load_file(self.data_dir / 'test' / 'performance-data.txt')
            
            # Load instance features
            train_features = self._load_file(self.data_dir / 'train' / 'instance-features.txt')
            test_features = self._load_file(self.data_dir / 'test' / 'instance-features.txt')
            
            # Validate data consistency
            self._validate_data_consistency(train_perf, train_features, test_perf, test_features)
            
            # Load metadata 
            self._load_metadata()
            
            logger.info(f"Data loaded successfully: {train_features.shape[0]} train instances, "
                f"{test_features.shape[0]} test instances, "
                f"{train_features.shape[1]} features, "
                f"{train_perf.shape[1]} algorithms")
            
            return train_perf, train_features, test_perf, test_features
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _load_file(self, filepath: Path) -> np.ndarray:
        """Load a single data file with format detection."""
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        suffix = filepath.suffix.lower()
        
        if suffix == '.txt' or suffix == '.csv':
            data = np.genfromtxt(filepath, delimiter=',')
        elif suffix == '.npy':
            data = np.load(filepath)
        elif suffix == '.npz':
            with np.load(filepath) as f:
                data = f['data']
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
            
        return data
    
    def _validate_data_consistency(self, train_perf: np.ndarray, train_features: np.ndarray,
            test_perf: np.ndarray, test_features: np.ndarray):
        """Validate that loaded data has consistent shapes."""
        if train_perf.shape[0] != train_features.shape[0]:
            raise ValueError("Training performance and features have different number of instances")
        if test_perf.shape[0] != test_features.shape[0]:
            raise ValueError("Test performance and features have different number of instances")
        if train_perf.shape[1] != test_perf.shape[1]:
            raise ValueError("Training and test sets have different number of algorithms")
        if train_features.shape[1] != test_features.shape[1]:
            raise ValueError("Training and test sets have different number of features")
    
    def _load_metadata(self):
        """Load metadata about features and algorithms if available."""
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names')
                self.algorithm_names = metadata.get('algorithm_names')
                logger.info("Metadata loaded successfully")
    
    def preprocess_features(self, train_features: np.ndarray, test_features: np.ndarray,
            scaler_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features with various scaling options."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        if scaler_type not in scalers:
            raise ValueError(f"Unknown scaler type: {scaler_type}. Choose from {list(scalers.keys())}")
        
        self.scaler = scalers[scaler_type]
        train_scaled = self.scaler.fit_transform(train_features)
        test_scaled = self.scaler.transform(test_features)
        
        logger.info(f"Features scaled using {scaler_type} scaler")
        return train_scaled, test_scaled
    
    def create_validation_split(self, train_features: np.ndarray, train_performance: np.ndarray,
            val_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """Create a validation split from training data."""
        X_train, X_val, y_train, y_val = train_test_split(
            train_features, train_performance, test_size=val_size, random_state=random_state
        )
        logger.info(f"Created validation split: {X_val.shape[0]} validation instances")
        return X_train, X_val, y_train, y_val