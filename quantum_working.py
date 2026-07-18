"""
Working Quantum ADR Demo
Fixed Qiskit v2.3+ compatibility
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Try quantum imports
try:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.algorithms.classifiers import VQC
    from qiskit_algorithms.optimizers import COBYLA
    QISKIT_AVAILABLE = True
    print("✅ Qiskit available")
except ImportError as e:
    QISKIT_AVAILABLE = False
    print(f"❌ Qiskit not available: {e}")

def prepare_dataset():
    """Prepare ADR dataset"""
    print("📊 Preparing ADR dataset...")
    
    # Features: [age, dosage, drug_count, condition_count]
    X = np.array([
        [25, 50, 1, 0],   # Young, low risk
        [60, 100, 3, 2],  # Elderly, high risk
        [30, 20, 2, 1],  # Middle-aged, medium risk
        [50, 80, 4, 1],  # Older, high risk
        [35, 60, 1, 0],  # Middle-aged, low-medium risk
        [70, 120, 5, 3], # Very elderly, very high risk
        [45, 40, 2, 1],  # Middle-aged, medium-high risk
        [55, 90, 3, 2],  # Older patient, high risk
        [40, 30, 1, 0],  # Middle-aged, low risk
        [65, 85, 2, 1],  # Elderly, medium-high risk
    ])
    
    y = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1])  # ADR occurrence
    
    # Normalize for quantum (0 to π)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_normalized = scaler.fit_transform(X)
    
    print(f"✅ Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 ADR rate: {np.mean(y)*100:.1f}%")
    
    return X_normalized, y, scaler

def create_quantum_classifier():
    """Create quantum classifier with Qiskit v2.3+"""
    print("⚛️ Creating quantum classifier...")
    
    if not QISKIT_AVAILABLE:
        return None
    
    try:
        # Using new Qiskit v2.3+ syntax
        feature_map = ZZFeatureMap(feature_dimension=4, reps=2)
        ansatz = RealAmplitudes(num_qubits=4, reps=2)
        
        # Create VQC with optimizer
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=50)
        )
        
        print("✅ Quantum VQC created successfully")
        return vqc
        
    except Exception as e:
        print(f"❌ Quantum classifier creation failed: {e}")
        return None

def train_models(X_train, y_train):
    """Train quantum and classical models"""
    print("🤖 Training models...")
    
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
        classical_model = RandomForestClassifier(n_estimators=100, random_state=42)
        classical_model.fit(X_train, y_train)
        models['classical'] = classical_model
        print("✅ Classical training completed!")
    except Exception as e:
        print(f"❌ Classical training failed: {e}")
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate both models"""
    print("🧪 Evaluating models...")
    
    results = {}
    
    for model_name, model in models.items():
        try:
            score = model.score(X_test, y_test)
            predictions = model.predict(X_test)
            
            results[model_name] = {
                'accuracy': score,
                'predictions': predictions
            }
            
            print(f"✅ {model_name.title()} accuracy: {score:.4f}")
            
        except Exception as e:
            print(f"❌ {model_name.title()} evaluation failed: {e}")
    
    return results

def hybrid_prediction(models, features, scaler):
    """Hybrid prediction combining both models"""
    print("🔗 Making hybrid prediction...")
    
    # Normalize features
    features_norm = scaler.transform([features])
    
    # Get predictions
    quantum_pred = None
    classical_pred = None
    
    if 'quantum' in models:
        try:
            quantum_pred = int(models['quantum'].predict(features_norm)[0])
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
        "confidence": abs(quantum_pred - classical_pred) if quantum_pred is not None and classical_pred is not None else 0.5
    }
    
    print(f"🎯 Hybrid result: {result}")
    return result

def main():
    """Main quantum training demo"""
    print("🚀 QUANTUM ADR PREDICTION DEMO")
    print("=" * 50)
    
    # STEP 1: Prepare dataset
    X, y, scaler = prepare_dataset()
    
    # STEP 2: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"📊 Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # STEP 3: Train models
    models = train_models(X_train, y_train)
    
    # STEP 4: Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # STEP 5: Hybrid prediction demo
    print("\n🔗 HYBRID PREDICTION DEMO")
    print("-" * 30)
    
    # Test with sample patient
    sample_features = [45, 75, 2, 1]  # Middle-aged, medium-high risk
    hybrid_result = hybrid_prediction(models, sample_features, scaler)
    
    print("\n🎉 FINAL RESULTS:")
    for model_name, result in results.items():
        print(f"📊 {model_name.title()} accuracy: {result['accuracy']:.4f}")
    
    print(f"🔗 Hybrid prediction for sample: {hybrid_result}")
    
    # Feature importance (classical)
    if 'classical' in models:
        feature_names = ['age', 'dosage', 'drug_count', 'condition_count']
        print(f"\n📈 Classical Feature Importance:")
        for i, importance in enumerate(models['classical'].feature_importances_):
            print(f"  {feature_names[i]}: {importance:.4f}")
    
    print("\n✅ Quantum ADR Demo Complete!")
    print("🚀 System ready for integration!")
    
    # Integration function example
    print("\n🔗 INTEGRATION FUNCTION:")
    print("def predict_adr_risk(patient_features):")
    print("    # Normalize features")
    print("    features_norm = scaler.transform([patient_features])")
    print("    # Quantum prediction")
    print("    if 'quantum' in models:")
    print("        quantum_pred = models['quantum'].predict(features_norm)[0]")
    print("    # Classical prediction")
    print("    if 'classical' in models:")
    print("        classical_pred = models['classical'].predict([patient_features])[0]")
    print("    # Hybrid ensemble")
    print("    final = (quantum_pred + classical_pred) / 2")
    print("    return {'quantum': quantum_pred, 'classical': classical_pred, 'final': int(final > 0.5)}")

if __name__ == "__main__":
    main()
