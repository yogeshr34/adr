"""
Simplified Quantum Training Demo for ADR Prediction System
Step-by-step quantum machine learning with basic Qiskit
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import warnings

# Check for basic Qiskit availability
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.algorithms.classifiers import VQC
    from qiskit_algorithms.optimizers import COBYLA
    QISKIT_AVAILABLE = True
    print("✅ Basic Qiskit imported successfully")
except ImportError as e:
    QISKIT_AVAILABLE = False
    print(f"❌ Qiskit import failed: {e}")
    print("Using classical fallback...")


def prepare_adr_dataset():
    """Prepare ADR dataset for quantum/classical training"""
    print("📊 STEP 2: Prepare Dataset (ADR-style)")
    
    # Features: [age, dosage, drug_count, condition_count, interaction_score]
    X = np.array([
        [25, 50, 1, 0, 0.3],   # Young patient, low risk
        [60, 100, 3, 2, 0.8],  # Elderly patient, high risk
        [30, 20, 2, 1, 0.5],  # Middle-aged, medium risk
        [50, 80, 4, 1, 0.9],  # Older patient, high risk
        [35, 60, 1, 0, 0.4],  # Middle-aged, low-medium risk
        [70, 120, 5, 3, 0.95], # Very elderly, very high risk
        [45, 40, 2, 1, 0.6],  # Middle-aged, medium-high risk
        [55, 90, 3, 2, 0.7],  # Older patient, high risk
        [40, 30, 1, 0, 0.2],  # Middle-aged, low risk
    ])
    
    y = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])  # ADR occurrence
    
    # Feature names
    feature_names = ['age', 'dosage', 'drug_count', 'condition_count', 'interaction_score']
    
    # Normalize for quantum
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_normalized = scaler.fit_transform(X)
    
    print(f"✅ Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 ADR rate: {np.mean(y)*100:.1f}%")
    
    return X_normalized, y, scaler, feature_names


def create_quantum_classifier():
    """Create quantum classifier using VQC"""
    print("⚛️ STEP 3: Build Quantum Classifier")
    
    if not QISKIT_AVAILABLE:
        print("❌ Qiskit not available, using classical fallback")
        return None
    
    try:
        # Create VQC with 5 qubits (one for each feature)
        vqc = VQC(
            feature_map=ZZFeatureMap(feature_dimension=5, reps=2),
            ansatz=RealAmplitudes(num_qubits=5, reps=2),
            optimizer=COBYLA(maxiter=50),
            callback=lambda w, nfev: print(f"🔄 Iteration {nfev}: Loss = {w:.4f}")
        )
        
        print("✅ Quantum VQC classifier created")
        return vqc
        
    except Exception as e:
        print(f"❌ Quantum classifier creation failed: {e}")
        return None


def train_models(X_train, y_train):
    """Train both quantum and classical models"""
    print("🤖 STEP 4: Train Models")
    
    models = {}
    
    # Train quantum model
    quantum_model = create_quantum_classifier()
    if quantum_model is not None:
        try:
            print("🚀 Training quantum model...")
            quantum_model.fit(X_train, y_train)
            models['quantum'] = quantum_model
            print("✅ Quantum training completed!")
        except Exception as e:
            print(f"❌ Quantum training failed: {e}")
    
    # Train classical model
    try:
        print("🤖 Training classical model...")
        classical_model = RandomForestClassifier(n_estimators=50, random_state=42)
        classical_model.fit(X_train, y_train)
        models['classical'] = classical_model
        print("✅ Classical training completed!")
    except Exception as e:
        print(f"❌ Classical training failed: {e}")
    
    return models


def evaluate_models(models, X_test, y_test):
    """Evaluate both models"""
    print("🧪 STEP 5: Evaluate Models")
    
    results = {}
    
    for model_name, model in models.items():
        try:
            score = model.score(X_test, y_test)
            predictions = model.predict(X_test)
            
            results[model_name] = {
                'accuracy': score,
                'predictions': predictions
            }
            
            print(f"✅ {model_name.title()} Model Accuracy: {score:.4f}")
            
        except Exception as e:
            print(f"❌ {model_name.title()} evaluation failed: {e}")
            results[model_name] = {'accuracy': 0.0, 'predictions': []}
    
    return results


def hybrid_predict(models, features, scaler):
    """Hybrid prediction combining both models"""
    print("🔗 STEP 6: Hybrid Prediction")
    
    # Normalize features
    features_normalized = scaler.transform([features])
    
    # Get predictions
    quantum_pred = None
    classical_pred = None
    
    if 'quantum' in models:
        try:
            quantum_pred = int(models['quantum'].predict(features_normalized)[0])
            print(f"⚛️ Quantum prediction: {quantum_pred}")
        except Exception as e:
            print(f"❌ Quantum prediction failed: {e}")
    
    if 'classical' in models:
        try:
            classical_pred = int(models['classical'].predict([features])[0])
            print(f"🤖 Classical prediction: {classical_pred}")
        except Exception as e:
            print(f"❌ Classical prediction failed: {e}")
    
    # Hybrid ensemble
    if quantum_pred is not None and classical_pred is not None:
        final = (quantum_pred + classical_pred) / 2
        final_binary = int(final > 0.5)
    elif quantum_pred is not None:
        final_binary = quantum_pred
    elif classical_pred is not None:
        final_binary = classical_pred
    else:
        final_binary = 0
    
    result = {
        "quantum": quantum_pred,
        "classical": classical_pred,
        "final": final_binary,
        "confidence": abs(quantum_pred - classical_pred) if quantum_pred and classical_pred else 0.5
    }
    
    print(f"🎯 Hybrid result: {result}")
    return result


def main():
    """Main function demonstrating quantum training pipeline"""
    print("🚀 QUANTUM ADR PREDICTION SYSTEM DEMO")
    print("=" * 50)
    
    # STEP 1: Check dependencies
    print("⚛️ STEP 1: Check Dependencies")
    if QISKIT_AVAILABLE:
        print("✅ Qiskit available")
    else:
        print("⚠️ Qiskit not fully available, using fallback")
    
    # STEP 2: Prepare dataset
    X, y, scaler, feature_names = prepare_adr_dataset()
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"📊 Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # STEP 3-4: Train models
    models = train_models(X_train, y_train)
    
    # STEP 5: Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # STEP 6: Demonstrate hybrid prediction
    print("\n🔗 STEP 6: Hybrid Prediction Demo")
    print("-" * 30)
    
    # Test with sample data
    sample_features = [45, 75, 3, 2, 0.7]  # Middle-aged, medium-high risk
    hybrid_result = hybrid_predict(models, sample_features, scaler)
    
    print("\n🎉 FINAL RESULTS:")
    for model_name, result in results.items():
        print(f"📊 {model_name.title()} accuracy: {result['accuracy']:.4f}")
    
    print(f"🔗 Hybrid prediction for sample: {hybrid_result}")
    
    # Feature importance (classical only)
    if 'classical' in models:
        print(f"\n📈 Classical Feature Importance:")
        for i, importance in enumerate(models['classical'].feature_importances_):
            print(f"  {feature_names[i]}: {importance:.4f}")
    
    # Quantum circuit info
    if 'quantum' in models and QISKIT_AVAILABLE:
        print(f"\n⚛️ Quantum Circuit Info:")
        print(f"  Feature map: {models['quantum'].feature_map}")
        print(f"  Ansatz: {models['quantum'].ansatz}")
        print(f"  Optimizer: {models['quantum'].optimizer}")
    
    print("\n✅ Quantum ADR Prediction Demo Complete!")
    print("🚀 Ready for integration with hybrid system!")
    
    # Integration example
    print("\n🔗 INTEGRATION EXAMPLE:")
    print("def quantum_predict_adr(features):")
    print("    # Normalize features")
    print("    features_norm = scaler.transform([features])")
    print("    # Quantum prediction")
    print("    quantum_pred = models['quantum'].predict(features_norm)[0]")
    print("    # Classical prediction")
    print("    classical_pred = models['classical'].predict([features])[0]")
    print("    # Hybrid ensemble")
    print("    final = (quantum_pred + classical_pred) / 2")
    print("    return int(final > 0.5)")


if __name__ == "__main__":
    main()
