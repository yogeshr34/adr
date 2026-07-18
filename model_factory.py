"""
Production-ready ADR Prediction System - Model Factory
Unified interface for multiple ML model backends with SHAP explainability
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import time
import warnings

# Import all model libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available. Install with: pip install catboost")

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    warnings.warn("TabNet not available. Install with: pip install pytorch-tabnet")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

try:
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Install with: pip install scikit-learn")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")


class BaseModelWrapper(BaseEstimator, ClassifierMixin, ABC):
    """Abstract base class for all model wrappers with sklearn compatibility"""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_ = None
        self.classes_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = None
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator (sklearn compatibility)"""
        return self.kwargs.copy()
    
    def set_params(self, **params) -> 'BaseModelWrapper':
        """Set parameters for this estimator (sklearn compatibility)"""
        for key, value in params.items():
            self.kwargs[key] = value
        return self
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance"""
        pass
    
    @abstractmethod
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Prepare data for the specific model"""
        pass
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'BaseModelWrapper':
        """Fit the model with sklearn-compatible interface"""
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=True, ensure_all_finite=False)
        
        # Store metadata
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        # Prepare data and fit
        X_prepared = self._prepare_data(X)
        self.model_ = self._create_model()
        
        if hasattr(self.model_, 'fit'):
            self.model_.fit(X_prepared, y)
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions with sklearn-compatible interface"""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=True, ensure_all_finite=False)
        X_prepared = self._prepare_data(X)
        
        if hasattr(self.model_, 'predict'):
            return self.model_.predict(X_prepared)
        else:
            raise NotImplementedError("Model does not implement predict method")
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make probability predictions with sklearn-compatible interface"""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=True, ensure_all_finite=False)
        X_prepared = self._prepare_data(X)
        
        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X_prepared)
        else:
            # For models without predict_proba, use decision_function or predict
            if hasattr(self.model_, 'decision_function'):
                decisions = self.model_.decision_function(X_prepared)
                if decisions.ndim == 1:
                    decisions = np.column_stack([-decisions, decisions])
                return (decisions - decisions.min()) / (decisions.max() - decisions.min())
            else:
                predictions = self.model_.predict(X_prepared).reshape(-1, 1)
                return np.column_stack([1 - predictions, predictions])
    
    def get_shap_explainer(self, X_background: Optional[np.ndarray] = None) -> Optional[Any]:
        """Get SHAP explainer for tree-based models"""
        if not SHAP_AVAILABLE:
            warnings.warn("SHAP not available")
            return None
        
        check_is_fitted(self)
        
        # Only tree-based models have efficient SHAP explainers
        if isinstance(self.model_, (xgb.XGBClassifier, CatBoostClassifier)) or \
           (SKLEARN_AVAILABLE and isinstance(self.model_, RandomForestClassifier)):
            
            if X_background is None:
                # Use training data if no background provided
                if hasattr(self, 'X_train_'):
                    X_background = self.X_train_
                else:
                    warnings.warn("No background data available for SHAP explainer")
                    return None
            
            if isinstance(self.model_, xgb.XGBClassifier):
                return shap.TreeExplainer(self.model_, data=X_background)
            elif isinstance(self.model_, CatBoostClassifier):
                return shap.TreeExplainer(self.model_)
            elif isinstance(self.model_, RandomForestClassifier):
                return shap.TreeExplainer(self.model_, data=X_background)
        
        return None


class XGBoostWrapper(BaseModelWrapper):
    """XGBoost model wrapper with sklearn compatibility"""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 6, subsample: float = 1.0, colsample_bytree: float = 1.0,
                 scale_pos_weight: Optional[float] = None, random_state: int = 42,
                 eval_metric: str = 'logloss', **kwargs):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric=eval_metric,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            **kwargs
        )
    
    def _create_model(self) -> xgb.XGBClassifier:
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        return xgb.XGBClassifier(**self.kwargs)
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        return X


class CatBoostWrapper(BaseModelWrapper):
    """CatBoost model wrapper with sklearn compatibility and categorical features support"""
    
    def __init__(self, iterations: int = 100, learning_rate: float = 0.1,
                 depth: int = 6, verbose: bool = False, cat_features: Optional[List[Union[int, str]]] = None,
                 random_state: int = 42, **kwargs):
        super().__init__(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            verbose=verbose,
            cat_features=cat_features,
            random_state=random_state,
            **kwargs
        )
        self.cat_features = cat_features
    
    def _create_model(self) -> CatBoostClassifier:
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available")
        
        return CatBoostClassifier(**self.kwargs)
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        # CatBoost handles categorical features automatically
        return X


class PyTorchMLPWrapper(BaseModelWrapper):
    """PyTorch MLP wrapper with sklearn compatibility"""
    
    def __init__(self, input_dim: Optional[int] = None, hidden_dims: List[int] = [128, 64],
                 dropout_rates: List[float] = [0.3, 0.2], learning_rate: float = 0.001,
                 epochs: int = 100, batch_size: int = 32, random_state: int = 42, **kwargs):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            **kwargs
        )
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
    
    def _create_model(self) -> 'ADRNet':
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        if self.input_dim is None:
            raise ValueError("input_dim must be specified for PyTorch MLP")
        
        return ADRNet(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rates=self.dropout_rates
        )
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame]) -> torch.Tensor:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return torch.FloatTensor(X)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'PyTorchMLPWrapper':
        """Fit PyTorch model with training loop"""
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=True, ensure_all_finite=False)
        
        # Store metadata
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        # Set input_dim if not specified
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        # Prepare data
        X_tensor = self._prepare_data(X)
        y_tensor = torch.LongTensor(y)
        
        # Create and train model
        self.model_ = self._create_model()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Store training data for potential SHAP use
        self.X_train_ = X
        
        # Training loop
        self.model_.train()
        for epoch in range(self.epochs):
            # Simple batch training
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                batch_y = y_tensor[i:i+self.batch_size].float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions with PyTorch model"""
        check_is_fitted(self)
        X_tensor = self._prepare_data(X)
        
        self.model_.eval()
        with torch.no_grad():
            probabilities = self.model_(X_tensor)
            predictions = (probabilities > 0.5).long().squeeze()
        
        return predictions.numpy()
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make probability predictions with PyTorch model"""
        check_is_fitted(self)
        X_tensor = self._prepare_data(X)
        
        self.model_.eval()
        with torch.no_grad():
            probabilities = self.model_(X_tensor)
            prob_numpy = probabilities.numpy()
        
        # Return [prob_class_0, prob_class_1] format
        return np.column_stack([1 - prob_numpy.squeeze(), prob_numpy.squeeze()])


class TabNetWrapper(BaseModelWrapper):
    """TabNet model wrapper with sklearn compatibility"""
    
    def __init__(self, n_steps: int = 3, gamma: float = 1.3, n_shared: int = 2,
                 n_independent: int = 2, verbose: int = 0, **kwargs):
        super().__init__(
            n_steps=n_steps,
            gamma=gamma,
            n_shared=n_shared,
            n_independent=n_independent,
            verbose=verbose,
            **kwargs
        )
    
    def _create_model(self) -> TabNetClassifier:
        if not TABNET_AVAILABLE:
            raise ImportError("TabNet not available")
        
        return TabNetClassifier(**self.kwargs)
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X


class RandomForestWrapper(BaseModelWrapper):
    """Random Forest model wrapper with sklearn compatibility"""
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 10,
                 min_samples_leaf: int = 5, class_weight: str = 'balanced',
                 random_state: int = 42, **kwargs):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            **kwargs
        )
    
    def _create_model(self) -> RandomForestClassifier:
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available")
        
        return RandomForestClassifier(**self.kwargs)
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        return X


class LogisticRegressionWrapper(BaseModelWrapper):
    """Logistic Regression wrapper with ElasticNet regularization"""
    
    def __init__(self, penalty: str = 'elasticnet', solver: str = 'saga',
                 l1_ratio: float = 0.5, max_iter: int = 1000, class_weight: str = 'balanced',
                 random_state: int = 42, **kwargs):
        super().__init__(
            penalty=penalty,
            solver=solver,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            **kwargs
        )
    
    def _create_model(self) -> LogisticRegression:
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available")
        
        return LogisticRegression(**self.kwargs)
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        return X


class StackingEnsembleWrapper(BaseModelWrapper):
    """Stacking Ensemble wrapper with sklearn compatibility"""
    
    def __init__(self, base_estimators: Optional[List[Tuple[str, Any]]] = None,
                 final_estimator: Optional[Any] = None, cv: int = 5, **kwargs):
        super().__init__(
            base_estimators=base_estimators,
            final_estimator=final_estimator,
            cv=cv,
            **kwargs
        )
        
        # Default estimators if none provided
        if base_estimators is None:
            self.base_estimators = [
                ('rf', RandomForestWrapper(n_estimators=100, random_state=42)),
                ('xgb', XGBoostWrapper(n_estimators=100, random_state=42)),
                ('lr', LogisticRegressionWrapper(random_state=42))
            ]
        else:
            self.base_estimators = base_estimators
        
        if final_estimator is None:
            self.final_estimator = LogisticRegressionWrapper(random_state=42)
        else:
            self.final_estimator = final_estimator
    
    def _create_model(self) -> StackingClassifier:
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available")
        
        return StackingClassifier(
            estimators=self.base_estimators,
            final_estimator=self.final_estimator,
            cv=self.kwargs.get('cv', 5)
        )
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        return X


class ADRNet(nn.Module):
    """PyTorch Neural Network for ADR prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 dropout_rates: List[float] = [0.3, 0.2]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ModelFactory:
    """Factory class for creating model instances with unified interface"""
    
    _models = {
        'xgboost': XGBoostWrapper,
        'catboost': CatBoostWrapper,
        'mlp': PyTorchMLPWrapper,
        'tabnet': TabNetWrapper,
        'random_forest': RandomForestWrapper,
        'logistic_regression': LogisticRegressionWrapper,
        'stacking': StackingEnsembleWrapper
    }
    
    @classmethod
    def get_model(cls, model_name: str, **kwargs) -> BaseModelWrapper:
        """
        Get model instance by name
        
        Args:
            model_name: Name of the model ('xgboost', 'catboost', 'mlp', etc.)
            **kwargs: Model-specific parameters
            
        Returns:
            Model instance with sklearn-compatible interface
        """
        if model_name not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Model '{model_name}' not available. Available models: {available_models}")
        
        model_class = cls._models[model_name]
        return model_class(**kwargs)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all available model names"""
        return list(cls._models.keys())
    
    @classmethod
    def get_clinical_priority_models(cls, **kwargs) -> Dict[str, BaseModelWrapper]:
        """
        Get clinical priority models: XGBoost for performance + SHAP, and Logistic Regression as regulatory baseline
        
        Returns:
            Dictionary with 'performance' and 'regulatory' models
        """
        return {
            'performance': cls.get_model('xgboost', **kwargs),  # XGBoost + SHAP for explainability
            'regulatory': cls.get_model('logistic_regression', **kwargs)  # Regulatory baseline
        }


# Utility function for automatic model selection based on data characteristics
def recommend_model(X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
                   prioritize_explainability: bool = True) -> str:
    """
    Recommend the best model based on data characteristics
    
    Args:
        X: Feature matrix
        y: Target vector
        prioritize_explainability: Whether to prioritize explainable models
        
    Returns:
        Recommended model name
    """
    X_array = X.values if hasattr(X, 'values') else X
    
    # Data characteristics
    n_samples, n_features = X_array.shape
    n_classes = len(np.unique(y))
    
    # Check for categorical features
    if hasattr(X, 'dtypes'):
        n_categorical = sum(X.dtypes == 'object')
    else:
        n_categorical = 0
    
    # Recommendations based on characteristics
    if prioritize_explainability:
        if n_categorical > 0:
            return 'catboost'  # Handles categorical features well
        else:
            return 'xgboost'  # Best performance + SHAP explainability
    else:
        if n_samples < 1000:
            return 'logistic_regression'  # Simple, fast for small datasets
        elif n_features > 100:
            return 'tabnet'  # Good for high-dimensional data
        else:
            return 'stacking'  # Best overall performance


# Export main classes and functions
__all__ = [
    'ModelFactory',
    'BaseModelWrapper',
    'XGBoostWrapper',
    'CatBoostWrapper', 
    'PyTorchMLPWrapper',
    'TabNetWrapper',
    'RandomForestWrapper',
    'LogisticRegressionWrapper',
    'StackingEnsembleWrapper',
    'ADRNet',
    'recommend_model'
]
