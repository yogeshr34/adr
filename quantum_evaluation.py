"""
Quantum Model Evaluation and Validation for Hybrid ADR Prediction System
Provides comprehensive evaluation metrics for quantum-classical models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class QuantumModelEvaluator:
    """
    Comprehensive evaluator for quantum-classical hybrid models
    Provides statistical analysis and visualization
    """
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "plots"):
        """
        Initialize evaluator
        
        Args:
            save_plots: Whether to save evaluation plots
            plot_dir: Directory to save plots
        """
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        self.evaluation_results = {}
        
        if self.save_plots:
            import os
            os.makedirs(plot_dir, exist_ok=True)
    
    def evaluate_model_performance(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray, 
                               y_pred_proba: np.ndarray,
                               model_name: str = "model",
                               quantum_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive model performance evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            quantum_info: Additional quantum model information
            
        Returns:
            Dictionary with comprehensive metrics
        """
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary')
            recall = recall_score(y_true, y_pred, average='binary')
            f1 = f1_score(y_true, y_pred, average='binary')
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
            
            # ROC curve data
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
            
            # Precision-Recall curve data
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
            
            # Optimal thresholds
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            optimal_threshold = roc_thresholds[optimal_idx]
            
            # Compile results
            results = {
                "model_name": model_name,
                "basic_metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "roc_auc": roc_auc
                },
                "clinical_metrics": {
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "ppv": ppv,
                    "npv": npv
                },
                "confusion_matrix": {
                    "matrix": cm.tolist(),
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp)
                },
                "threshold_analysis": {
                    "optimal_threshold": float(optimal_threshold),
                    "youden_j": float(youden_j[optimal_idx]),
                    "roc_curve": {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "thresholds": roc_thresholds.tolist()
                    },
                    "pr_curve": {
                        "precision": precision_curve.tolist(),
                        "recall": recall_curve.tolist(),
                        "thresholds": pr_thresholds.tolist()
                    }
                }
            }
            
            # Add quantum-specific metrics if available
            if quantum_info:
                results["quantum_metrics"] = {
                    "quantum_mode": quantum_info.get("quantum_mode", "unknown"),
                    "confidence": quantum_info.get("coherence", 0),
                    "entanglement_strength": quantum_info.get("entanglement_strength", 0),
                    "n_qubits": quantum_info.get("n_qubits", 0),
                    "quantum_volume": quantum_info.get("quantum_volume", 0)
                }
            
            self.evaluation_results[model_name] = results
            logger.info(f"✅ Evaluated {model_name} - ROC-AUC: {roc_auc:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            return {"error": str(e)}
    
    def compare_models(self, 
                    model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models side-by-side
        
        Args:
            model_results: Dictionary of model evaluation results
            
        Returns:
            Comparison DataFrame
        """
        try:
            comparison_data = []
            
            for model_name, results in model_results.items():
                if "error" not in results:
                    basic_metrics = results.get("basic_metrics", {})
                    clinical_metrics = results.get("clinical_metrics", {})
                    quantum_metrics = results.get("quantum_metrics", {})
                    
                    row = {
                        "Model": model_name,
                        "Accuracy": basic_metrics.get("accuracy", 0),
                        "Precision": basic_metrics.get("precision", 0),
                        "Recall": basic_metrics.get("recall", 0),
                        "F1-Score": basic_metrics.get("f1_score", 0),
                        "ROC-AUC": basic_metrics.get("roc_auc", 0),
                        "Sensitivity": clinical_metrics.get("sensitivity", 0),
                        "Specificity": clinical_metrics.get("specificity", 0),
                        "PPV": clinical_metrics.get("ppv", 0),
                        "NPV": clinical_metrics.get("npv", 0)
                    }
                    
                    # Add quantum metrics if available
                    if quantum_metrics:
                        row["Quantum_Mode"] = quantum_metrics.get("quantum_mode", "N/A")
                        row["Quantum_Confidence"] = quantum_metrics.get("confidence", 0)
                        row["Entanglement"] = quantum_metrics.get("entanglement_strength", 0)
                    
                    comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values("ROC-AUC", ascending=False)
            
            self.evaluation_results["comparison_table"] = comparison_df
            
            logger.info(f"✅ Model comparison complete - {len(comparison_df)} models compared")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"❌ Model comparison failed: {e}")
            return pd.DataFrame()
    
    def statistical_significance_test(self,
                                 y_true: np.ndarray,
                                 model1_probs: np.ndarray,
                                 model2_probs: np.ndarray,
                                 model1_name: str = "Model 1",
                                 model2_name: str = "Model 2") -> Dict[str, Any]:
        """
        Perform statistical significance test between two models
        
        Args:
            y_true: True labels
            model1_probs: Probabilities from model 1
            model2_probs: Probabilities from model 2
            model1_name: Name of model 1
            model2_name: Name of model 2
            
        Returns:
            Statistical test results
        """
        try:
            from scipy import stats
            
            # Calculate ROC-AUC for both models
            auc1 = roc_auc_score(y_true, model1_probs)
            auc2 = roc_auc_score(y_true, model2_probs)
            
            # DeLong's test for ROC-AUC comparison
            # Simplified version using bootstrap
            n_bootstrap = 1000
            auc_diffs = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(y_true), len(y_true), replace=True)
                y_boot = y_true[indices]
                probs1_boot = model1_probs[indices]
                probs2_boot = model2_probs[indices]
                
                # Calculate AUC difference
                auc1_boot = roc_auc_score(y_boot, probs1_boot)
                auc2_boot = roc_auc_score(y_boot, probs2_boot)
                auc_diffs.append(auc1_boot - auc2_boot)
            
            # Calculate p-value
            auc_diff_observed = auc1 - auc2
            p_value = np.mean(np.abs(auc_diffs) >= np.abs(auc_diff_observed))
            
            # Confidence interval for difference
            ci_lower = np.percentile(auc_diffs, 2.5)
            ci_upper = np.percentile(auc_diffs, 97.5)
            
            results = {
                "model1": {
                    "name": model1_name,
                    "auc": auc1
                },
                "model2": {
                    "name": model2_name,
                    "auc": auc2
                },
                "difference": {
                    "auc_diff": auc_diff_observed,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                },
                "test_info": {
                    "n_bootstrap": n_bootstrap,
                    "test_type": "bootstrap_delong"
                }
            }
            
            logger.info(f"✅ Statistical test: {model1_name} vs {model2_name} - p-value: {p_value:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Statistical test failed: {e}")
            return {"error": str(e)}
    
    def plot_roc_curves(self, 
                        model_results: Dict[str, Dict[str, Any]],
                        save_name: str = "roc_curves") -> None:
        """
        Plot ROC curves for multiple models
        
        Args:
            model_results: Dictionary of model evaluation results
            save_name: Name for saved plot
        """
        try:
            plt.figure(figsize=(10, 8))
            
            for model_name, results in model_results.items():
                if "error" not in results and "threshold_analysis" in results:
                    roc_data = results["threshold_analysis"]["roc_curve"]
                    fpr = roc_data["fpr"]
                    tpr = roc_data["tpr"]
                    auc = results["basic_metrics"]["roc_auc"]
                    
                    plt.plot(fpr, tpr, 
                            label=f'{model_name} (AUC = {auc:.3f})',
                            linewidth=2)
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves - Model Comparison')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            if self.save_plots:
                plt.savefig(f"{self.plot_dir}/{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                logger.info(f"✅ Saved ROC curves to {self.plot_dir}/{save_name}.png")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ ROC curve plotting failed: {e}")
    
    def plot_precision_recall_curves(self,
                                  model_results: Dict[str, Dict[str, Any]],
                                  save_name: str = "pr_curves") -> None:
        """
        Plot Precision-Recall curves for multiple models
        
        Args:
            model_results: Dictionary of model evaluation results
            save_name: Name for saved plot
        """
        try:
            plt.figure(figsize=(10, 8))
            
            for model_name, results in model_results.items():
                if "error" not in results and "threshold_analysis" in results:
                    pr_data = results["threshold_analysis"]["pr_curve"]
                    precision = pr_data["precision"]
                    recall = pr_data["recall"]
                    auc = results["basic_metrics"]["roc_auc"]
                    
                    plt.plot(recall, precision,
                            label=f'{model_name} (AUC = {auc:.3f})',
                            linewidth=2)
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves - Model Comparison')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            if self.save_plots:
                plt.savefig(f"{self.plot_dir}/{save_name}.png",
                           dpi=300, bbox_inches='tight')
                logger.info(f"✅ Saved PR curves to {self.plot_dir}/{save_name}.png")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ PR curve plotting failed: {e}")
    
    def plot_confusion_matrices(self,
                             model_results: Dict[str, Dict[str, Any]],
                             save_name: str = "confusion_matrices") -> None:
        """
        Plot confusion matrices for multiple models
        
        Args:
            model_results: Dictionary of model evaluation results
            save_name: Name for saved plot
        """
        try:
            n_models = len(model_results)
            fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
            
            if n_models == 1:
                axes = [axes]
            
            for idx, (model_name, results) in enumerate(model_results.items()):
                if "error" not in results:
                    cm = results["confusion_matrix"]["matrix"]
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                              ax=axes[idx], cbar=False)
                    axes[idx].set_title(f'{model_name}\nAUC = {results["basic_metrics"]["roc_auc"]:.3f}')
                    axes[idx].set_xlabel('Predicted')
                    axes[idx].set_ylabel('Actual')
            
            plt.tight_layout()
            
            if self.save_plots:
                plt.savefig(f"{self.plot_dir}/{save_name}.png",
                           dpi=300, bbox_inches='tight')
                logger.info(f"✅ Saved confusion matrices to {self.plot_dir}/{save_name}.png")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Confusion matrix plotting failed: {e}")
    
    def plot_quantum_analysis(self,
                           model_results: Dict[str, Dict[str, Any]],
                           save_name: str = "quantum_analysis") -> None:
        """
        Plot quantum-specific analysis
        
        Args:
            model_results: Dictionary of model evaluation results
            save_name: Name for saved plot
        """
        try:
            # Extract quantum models only
            quantum_models = {k: v for k, v in model_results.items() 
                           if "quantum_metrics" in v}
            
            if not quantum_models:
                logger.info("No quantum models found for analysis")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            model_names = list(quantum_models.keys())
            confidences = [quantum_models[name]["quantum_metrics"]["confidence"] 
                          for name in model_names]
            entanglements = [quantum_models[name]["quantum_metrics"]["entanglement_strength"]
                            for name in model_names]
            roc_aucs = [quantum_models[name]["basic_metrics"]["roc_auc"]
                        for name in model_names]
            
            # Confidence vs Performance
            axes[0, 0].scatter(confidences, roc_aucs, s=100, alpha=0.7)
            axes[0, 0].set_xlabel('Quantum Confidence')
            axes[0, 0].set_ylabel('ROC-AUC')
            axes[0, 0].set_title('Quantum Confidence vs Performance')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Entanglement vs Performance
            axes[0, 1].scatter(entanglements, roc_aucs, s=100, alpha=0.7)
            axes[0, 1].set_xlabel('Entanglement Strength')
            axes[0, 1].set_ylabel('ROC-AUC')
            axes[0, 1].set_title('Entanglement vs Performance')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Model comparison
            x_pos = np.arange(len(model_names))
            axes[1, 0].bar(x_pos, roc_aucs, alpha=0.7)
            axes[1, 0].set_xlabel('Quantum Model')
            axes[1, 0].set_ylabel('ROC-AUC')
            axes[1, 0].set_title('Quantum Model Performance')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(model_names, rotation=45)
            
            # Quantum modes
            modes = [quantum_models[name]["quantum_metrics"]["quantum_mode"]
                    for name in model_names]
            unique_modes = list(set(modes))
            mode_counts = [modes.count(mode) for mode in unique_modes]
            
            axes[1, 1].pie(mode_counts, labels=unique_modes, autopct='%1.1f%%')
            axes[1, 1].set_title('Quantum Backend Distribution')
            
            plt.tight_layout()
            
            if self.save_plots:
                plt.savefig(f"{self.plot_dir}/{save_name}.png",
                           dpi=300, bbox_inches='tight')
                logger.info(f"✅ Saved quantum analysis to {self.plot_dir}/{save_name}.png")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Quantum analysis plotting failed: {e}")
    
    def generate_evaluation_report(self,
                              model_results: Dict[str, Dict[str, Any]],
                              save_path: str = "evaluation_report.json") -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            model_results: Dictionary of model evaluation results
            save_path: Path to save report
            
        Returns:
            Comprehensive evaluation report
        """
        try:
            # Create comparison table
            comparison_df = self.compare_models(model_results)
            
            # Generate report
            report = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_models": len(model_results),
                    "successful_evaluations": len([r for r in model_results.values() if "error" not in r]),
                    "best_model": comparison_df.iloc[0]["Model"] if len(comparison_df) > 0 else None,
                    "best_roc_auc": comparison_df.iloc[0]["ROC-AUC"] if len(comparison_df) > 0 else 0,
                    "performance_range": {
                        "min_auc": comparison_df["ROC-AUC"].min() if len(comparison_df) > 0 else 0,
                        "max_auc": comparison_df["ROC-AUC"].max() if len(comparison_df) > 0 else 0,
                        "mean_auc": comparison_df["ROC-AUC"].mean() if len(comparison_df) > 0 else 0
                    }
                },
                "model_details": model_results,
                "comparison_table": comparison_df.to_dict('records'),
                "recommendations": self._generate_recommendations(comparison_df)
            }
            
            # Save report
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"✅ Evaluation report saved to {save_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, comparison_df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        if len(comparison_df) == 0:
            return ["No models available for recommendation"]
        
        best_model = comparison_df.iloc[0]
        best_auc = best_model["ROC-AUC"]
        
        # Performance recommendations
        if best_auc > 0.9:
            recommendations.append("Excellent performance achieved (ROC-AUC > 0.9)")
        elif best_auc > 0.8:
            recommendations.append("Good performance achieved (ROC-AUC > 0.8)")
        elif best_auc > 0.7:
            recommendations.append("Moderate performance achieved (ROC-AUC > 0.7)")
        else:
            recommendations.append("Performance needs improvement (ROC-AUC < 0.7)")
        
        # Model-specific recommendations
        if "Quantum_Mode" in best_model and pd.notna(best_model["Quantum_Mode"]):
            if best_model["Quantum_Confidence"] > 0.8:
                recommendations.append("Quantum model shows high confidence - consider quantum hardware deployment")
            else:
                recommendations.append("Quantum model confidence is moderate - consider parameter tuning")
        
        # Ensemble recommendations
        if len(comparison_df) > 2:
            recommendations.append("Consider ensemble methods to combine strengths of multiple models")
        
        # Clinical recommendations
        if best_model["Sensitivity"] > 0.8 and best_model["Specificity"] > 0.8:
            recommendations.append("Model meets clinical thresholds for both sensitivity and specificity")
        elif best_model["Sensitivity"] < 0.8:
            recommendations.append("Consider improving sensitivity for clinical safety")
        elif best_model["Specificity"] < 0.8:
            recommendations.append("Consider improving specificity for clinical efficiency")
        
        return recommendations


# Factory function
def create_quantum_evaluator(save_plots: bool = True, 
                           plot_dir: str = "plots") -> QuantumModelEvaluator:
    """Factory function to create quantum evaluator"""
    return QuantumModelEvaluator(save_plots, plot_dir)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.randint(0, 2, n_samples)
    
    # Simulate model predictions
    model_results = {
        "Classical_RF": {
            "basic_metrics": {"roc_auc": 0.85, "accuracy": 0.78, "precision": 0.76, "recall": 0.82, "f1_score": 0.79},
            "clinical_metrics": {"sensitivity": 0.82, "specificity": 0.74, "ppv": 0.76, "npv": 0.80},
            "confusion_matrix": {"matrix": [[370, 130], [90, 410]], "true_negatives": 370, "false_positives": 130, "false_negatives": 90, "true_positives": 410},
            "threshold_analysis": {
                "roc_curve": {
                    "fpr": [0, 0.26, 1], "tpr": [0, 0.82, 1], "thresholds": [2, 1, 0]
                }
            }
        },
        "Quantum_VQC": {
            "basic_metrics": {"roc_auc": 0.89, "accuracy": 0.82, "precision": 0.80, "recall": 0.85, "f1_score": 0.82},
            "clinical_metrics": {"sensitivity": 0.85, "specificity": 0.79, "ppv": 0.80, "npv": 0.84},
            "confusion_matrix": {"matrix": [[395, 105], [75, 425]], "true_negatives": 395, "false_positives": 105, "false_negatives": 75, "true_positives": 425},
            "threshold_analysis": {
                "roc_curve": {
                    "fpr": [0, 0.21, 1], "tpr": [0, 0.85, 1], "thresholds": [2, 1, 0]
                }
            },
            "quantum_metrics": {
                "quantum_mode": "qiskit_simulation",
                "confidence": 0.87,
                "entanglement_strength": 0.73,
                "n_qubits": 4
            }
        }
    }
    
    # Create evaluator
    evaluator = create_quantum_evaluator()
    
    # Generate comparison
    comparison = evaluator.compare_models(model_results)
    print("Model Comparison:")
    print(comparison)
    
    # Generate report
    report = evaluator.generate_evaluation_report(model_results)
    print("\nEvaluation Report Generated")
    print(f"Best Model: {report['summary']['best_model']}")
    print(f"Best ROC-AUC: {report['summary']['best_roc_auc']:.4f}")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")
