"""
Model Evaluation Utilities for ADR Prediction System
Comprehensive model comparison with clinical metrics
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix
)
import time
import warnings

from model_factory import ModelFactory, BaseModelWrapper


class ModelEvaluator:
    """Model evaluation with clinical metrics"""
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        """Initialize model evaluator"""
        self.cv_folds = cv_folds
        self.random_state = random_state
    
    def evaluate_single_model(self, model_name: str, model: BaseModelWrapper,
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate a single model"""
        start_time = time.time()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        fit_time = time.time() - start_time
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Clinical metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=self.cv_folds, random_state=self.random_state),
            scoring='roc_auc'
        )
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'fit_time': fit_time,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def compare_models(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare multiple models"""
        if model_names is None:
            model_names = ['xgboost', 'catboost', 'lightgbm', 'random_forest', 
                        'logistic_regression', 'tabnet', 'stacking']
        
        results = []
        
        for model_name in model_names:
            try:
                model = ModelFactory.get_model(model_name)
                result = self.evaluate_single_model(model_name, model, 
                                               X_train, y_train, X_test, y_test)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Failed to evaluate {model_name}: {e}")
                continue
        
        df = pd.DataFrame(results)
        return df.sort_values('roc_auc', ascending=False)


def compare_models(X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 model_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Convenience function to compare models"""
    evaluator = ModelEvaluator()
    return evaluator.compare_models(X_train, y_train, X_test, y_test, model_names)
