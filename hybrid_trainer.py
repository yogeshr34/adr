"""
Hybrid Model Trainer for Quantum-Classical ADR Prediction System
Integrates classical ML models with quantum classifiers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import joblib
import json
from datetime import datetime
import os

# Import classical ML models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Import quantum module
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from quantum import create_quantum_classifier
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    logging.warning("Quantum module not available")

logger = logging.getLogger(__name__)


class HybridModelTrainer:
    """
    Hybrid trainer for classical and quantum models
    Supports ensemble training and evaluation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize hybrid model trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Model storage
        self.classical_models = {}
        self.quantum_model = None
        self.scaler = StandardScaler()
        
        # Training data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
        # Results storage
        self.training_results = {}
        self.evaluation_results = {}
        
        # Initialize models
        self._initialize_models()
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "classical_models": {
                "xgboost": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42
                },
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                }
            },
            "quantum": {
                "n_qubits": 4,
                "backend": "auto",
                "max_iter": 100
            },
            "ensemble": {
                "weights": {"xgboost": 0.4, "random_forest": 0.3, "quantum": 0.3},
                "method": "weighted_average"
            },
            "training": {
                "test_size": 0.2,
                "random_state": 42,
                "cv_folds": 5
            }
        }
    
    def _initialize_models(self):
        """Initialize all models"""
        try:
            # Initialize classical models
            for model_name, params in self.config["classical_models"].items():
                if model_name == "xgboost":
                    self.classical_models[model_name] = xgb.XGBClassifier(**params)
                elif model_name == "random_forest":
                    self.classical_models[model_name] = RandomForestClassifier(**params)
            
            logger.info(f"✅ Initialized {len(self.classical_models)} classical models")
            
            # Initialize quantum model
            if QUANTUM_AVAILABLE:
                quantum_config = self.config["quantum"]
                self.quantum_model = create_quantum_classifier(
                    n_qubits=quantum_config["n_qubits"],
                    backend=quantum_config["backend"]
                )
                logger.info("✅ Quantum model initialized")
            else:
                logger.warning("⚠️ Quantum model not available")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            raise
    
    def load_data(self, X: Union[np.ndarray, pd.DataFrame], 
                  y: Union[np.ndarray, pd.Series],
                  feature_names: Optional[List[str]] = None):
        """
        Load and prepare training data
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
        """
        try:
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                self.feature_names = feature_names or X.columns.tolist()
                X = X.values
            else:
                self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            
            if isinstance(y, pd.Series):
                y = y.values
            
            # Split data
            training_config = self.config["training"]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=training_config["test_size"],
                random_state=training_config["random_state"],
                stratify=y
            )
            
            # Scale features
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            logger.info(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
            logger.info(f"✅ Train set: {self.X_train.shape[0]}, Test set: {self.X_test.shape[0]}")
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            raise
    
    def train_classical_models(self) -> Dict[str, Any]:
        """Train all classical models"""
        results = {}
        
        for model_name, model in self.classical_models.items():
            try:
                logger.info(f"🔄 Training {model_name}...")
                
                # Train model
                model.fit(self.X_train_scaled, self.y_train)
                
                # Predictions
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                # Metrics
                metrics = {
                    "accuracy": accuracy_score(self.y_test, y_pred),
                    "roc_auc": roc_auc_score(self.y_test, y_pred_proba),
                    "classification_report": classification_report(self.y_test, y_pred),
                    "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist()
                }
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, self.X_train_scaled, self.y_train,
                    cv=self.config["training"]["cv_folds"],
                    scoring='roc_auc'
                )
                metrics["cv_mean"] = cv_scores.mean()
                metrics["cv_std"] = cv_scores.std()
                
                results[model_name] = {
                    "model": model,
                    "metrics": metrics,
                    "predictions": y_pred,
                    "probabilities": y_pred_proba
                }
                
                logger.info(f"✅ {model_name} trained - ROC-AUC: {metrics['roc_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ {model_name} training failed: {e}")
                results[model_name] = {"error": str(e)}
        
        self.training_results.update(results)
        return results
    
    def train_quantum_model(self) -> Dict[str, Any]:
        """Train quantum model"""
        if not QUANTUM_AVAILABLE or self.quantum_model is None:
            return {"error": "Quantum model not available"}
        
        try:
            logger.info("🔄 Training quantum model...")
            
            # Train quantum model
            self.quantum_model.fit(self.X_train_scaled, self.y_train)
            
            # Predictions
            y_pred = self.quantum_model.predict(self.X_test_scaled)
            y_pred_proba = self.quantum_model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Metrics
            metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "roc_auc": roc_auc_score(self.y_test, y_pred_proba),
                "classification_report": classification_report(self.y_test, y_pred),
                "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist()
            }
            
            # Quantum-specific metrics
            quantum_info = self.quantum_model.get_quantum_state_info(self.X_test_scaled)
            metrics.update(quantum_info)
            
            result = {
                "model": self.quantum_model,
                "metrics": metrics,
                "predictions": y_pred,
                "probabilities": y_pred_proba,
                "quantum_info": quantum_info
            }
            
            self.training_results["quantum"] = result
            logger.info(f"✅ Quantum model trained - ROC-AUC: {metrics['roc_auc']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Quantum training failed: {e}")
            return {"error": str(e)}
    
    def train_ensemble(self) -> Dict[str, Any]:
        """Train ensemble of all models"""
        logger.info("🔄 Training hybrid ensemble...")
        
        # Train all individual models
        classical_results = self.train_classical_models()
        quantum_result = self.train_quantum_model()
        
        # Create ensemble predictions
        ensemble_predictions = self._create_ensemble_predictions()
        
        # Evaluate ensemble
        ensemble_metrics = {
            "accuracy": accuracy_score(self.y_test, ensemble_predictions["predictions"]),
            "roc_auc": roc_auc_score(self.y_test, ensemble_predictions["probabilities"]),
            "classification_report": classification_report(self.y_test, ensemble_predictions["predictions"]),
            "confusion_matrix": confusion_matrix(self.y_test, ensemble_predictions["predictions"]).tolist()
        }
        
        ensemble_result = {
            "predictions": ensemble_predictions["predictions"],
            "probabilities": ensemble_predictions["probabilities"],
            "metrics": ensemble_metrics,
            "individual_results": {**classical_results, "quantum": quantum_result}
        }
        
        self.training_results["ensemble"] = ensemble_result
        logger.info(f"✅ Ensemble trained - ROC-AUC: {ensemble_metrics['roc_auc']:.4f}")
        
        return ensemble_result
    
    def _create_ensemble_predictions(self) -> Dict[str, np.ndarray]:
        """Create ensemble predictions from individual models"""
        predictions = []
        probabilities = []
        weights = self.config["ensemble"]["weights"]
        
        # Collect predictions from all available models
        for model_name in ["xgboost", "random_forest", "quantum"]:
            if model_name in self.training_results and "error" not in self.training_results[model_name]:
                model_result = self.training_results[model_name]
                predictions.append(model_result["predictions"])
                probabilities.append(model_result["probabilities"])
        
        if not predictions:
            raise ValueError("No trained models available for ensemble")
        
        # Weighted average for probabilities
        ensemble_prob = np.zeros_like(probabilities[0])
        total_weight = 0
        
        for i, model_name in enumerate(["xgboost", "random_forest", "quantum"]):
            if model_name in weights and i < len(probabilities):
                ensemble_prob += weights[model_name] * probabilities[i]
                total_weight += weights[model_name]
        
        if total_weight > 0:
            ensemble_prob /= total_weight
        
        # Convert probabilities to predictions
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        return {
            "predictions": ensemble_pred,
            "probabilities": ensemble_prob
        }
    
    def evaluate_models(self) -> Dict[str, Any]:
        """Comprehensive evaluation of all models"""
        evaluation = {}
        
        for model_name, result in self.training_results.items():
            if "error" not in result:
                metrics = result["metrics"]
                
                evaluation[model_name] = {
                    "roc_auc": metrics["roc_auc"],
                    "accuracy": metrics["accuracy"],
                    "cv_mean": metrics.get("cv_mean", 0),
                    "cv_std": metrics.get("cv_std", 0)
                }
                
                # Add quantum-specific metrics
                if model_name == "quantum":
                    evaluation[model_name].update({
                        "quantum_mode": metrics.get("quantum_mode", "unknown"),
                        "confidence": metrics.get("coherence", 0),
                        "entanglement_strength": metrics.get("entanglement_strength", 0)
                    })
        
        # Create comparison table
        comparison_df = pd.DataFrame(evaluation).T
        comparison_df = comparison_df.sort_values("roc_auc", ascending=False)
        
        self.evaluation_results = {
            "individual_metrics": evaluation,
            "comparison_table": comparison_df,
            "best_model": comparison_df.index[0],
            "best_score": comparison_df.iloc[0]["roc_auc"]
        }
        
        logger.info(f"✅ Best model: {self.evaluation_results['best_model']} (ROC-AUC: {self.evaluation_results['best_score']:.4f})")
        
        return self.evaluation_results
    
    def save_models(self, save_dir: str = "models") -> Dict[str, str]:
        """Save all trained models"""
        os.makedirs(save_dir, exist_ok=True)
        saved_files = {}
        
        try:
            # Save classical models
            for model_name, model in self.classical_models.items():
                if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                    filename = os.path.join(save_dir, f"{model_name}_model.pkl")
                    joblib.dump(model, filename)
                    saved_files[model_name] = filename
                    logger.info(f"✅ Saved {model_name} model")
            
            # Save quantum model
            if self.quantum_model and self.quantum_model.is_trained:
                quantum_filename = os.path.join(save_dir, "quantum_model.pkl")
                joblib.dump(self.quantum_model, quantum_filename)
                saved_files["quantum"] = quantum_filename
                logger.info("✅ Saved quantum model")
            
            # Save scaler
            scaler_filename = os.path.join(save_dir, "scaler.pkl")
            joblib.dump(self.scaler, scaler_filename)
            saved_files["scaler"] = scaler_filename
            
            # Save results
            results_filename = os.path.join(save_dir, "training_results.json")
            with open(results_filename, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = self._make_json_serializable(self.training_results)
                json.dump(serializable_results, f, indent=2, default=str)
            saved_files["results"] = results_filename
            
            # Save config
            config_filename = os.path.join(save_dir, "config.json")
            with open(config_filename, 'w') as f:
                json.dump(self.config, f, indent=2)
            saved_files["config"] = config_filename
            
            logger.info(f"✅ All models saved to {save_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"❌ Model saving failed: {e}")
            return {"error": str(e)}
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other objects to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "data_info": {
                "n_samples": len(self.X_train) + len(self.X_test) if self.X_train is not None else 0,
                "n_features": len(self.feature_names) if self.feature_names else 0,
                "n_train": len(self.X_train) if self.X_train is not None else 0,
                "n_test": len(self.X_test) if self.X_test is not None else 0
            },
            "models_trained": list(self.training_results.keys()),
            "quantum_available": QUANTUM_AVAILABLE,
            "evaluation": self.evaluation_results
        }
        
        return summary


# Factory function
def create_hybrid_trainer(config: Optional[Dict] = None) -> HybridModelTrainer:
    """Factory function to create hybrid trainer"""
    return HybridModelTrainer(config)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create and train hybrid model
    trainer = create_hybrid_trainer()
    trainer.load_data(X, y, feature_names)
    
    # Train all models
    results = trainer.train_ensemble()
    
    # Evaluate models
    evaluation = trainer.evaluate_models()
    
    # Save models
    saved_files = trainer.save_models("hybrid_models")
    
    # Print summary
    summary = trainer.get_training_summary()
    print("✅ Hybrid Training Complete")
    print(f"Best model: {evaluation['best_model']}")
    print(f"Best score: {evaluation['best_score']:.4f}")
    print(f"Models saved: {list(saved_files.keys())}")
