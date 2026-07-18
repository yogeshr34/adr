"""
Quantum Training Demo for ADR Prediction System
Step-by-step quantum machine learning implementation
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import warnings

# Check for Qiskit availability
try:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit.primitives import Estimator
    from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_ibm_runtime import QiskitRuntimeService
    QISKIT_AVAILABLE = True
    print("✅ Qiskit imported successfully")
except ImportError as e:
    QISKIT_AVAILABLE = False
    print(f"❌ Qiskit import failed: {e}")
    print("Please install with: pip install qiskit qiskit-machine-learning qiskit-ibm-runtime")


def prepare_adr_dataset():
    """
    Prepare ADR-style dataset for quantum training
    
    Returns:
        X: Normalized features
        y: Labels
        scaler: Fitted scaler for future use
        feature_names: List of feature names
    """
    print("📊 STEP 2: Prepare Dataset (ADR-style)")
    
    # Example ADR dataset with realistic features
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
    
    # Feature names for explainability
    feature_names = ['age', 'dosage', 'drug_count', 'condition_count', 'interaction_score']
    
    # Normalize (VERY IMPORTANT for quantum)
    print("🔧 Normalizing features for quantum encoding...")
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_normalized = scaler.fit_transform(X)
    
    print(f"✅ Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 Feature distribution: {np.mean(y, axis=0):.2%} ADR rate")
    
    return X_normalized, y, scaler, feature_names


def build_quantum_circuit(num_features):
    """
    Build quantum feature map and ansatz
    
    Args:
        num_features: Number of input features
        
    Returns:
        feature_map: Quantum feature map
        ansatz: Variational circuit
    """
    print("⚛️ STEP 3: Build Quantum Feature Map")
    
    if not QISKIT_AVAILABLE:
        print("❌ Qiskit not available, skipping quantum circuit build")
        return None, None
    
    try:
        # ZZFeatureMap for encoding classical data
        feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
        print(f"✅ ZZFeatureMap created with {num_features} features, 2 repetitions")
        
        # RealAmplitudes ansatz for variational quantum circuit
        ansatz = RealAmplitudes(num_qubits=num_features, reps=2)
        print(f"✅ RealAmplitudes ansatz created with {num_features} qubits, 2 repetitions")
        
        return feature_map, ansatz
        
    except Exception as e:
        print(f"❌ Quantum circuit build failed: {e}")
        return None, None


def create_quantum_neural_network(feature_map, ansatz):
    """
    Create quantum neural network
    
    Args:
        feature_map: Quantum feature map
        ansatz: Variational circuit
        
    Returns:
        qnn: Quantum neural network
        estimator: Qiskit estimator
    """
    print("🧠 STEP 5: Combine → Quantum Neural Network")
    
    if not QISKIT_AVAILABLE or feature_map is None or ansatz is None:
        print("❌ Cannot create quantum neural network - missing components")
        return None, None
    
    try:
        # Create estimator
        estimator = Estimator()
        
        # Create quantum neural network
        qnn = EstimatorQNN(
            estimator=estimator,
            circuit=feature_map.compose(ansatz),
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )
        
        print(f"✅ Quantum Neural Network created")
        print(f"📊 Input parameters: {len(feature_map.parameters)}")
        print(f"📊 Weight parameters: {len(ansatz.parameters)}")
        
        return qnn, estimator
        
    except Exception as e:
        print(f"❌ Quantum neural network creation failed: {e}")
        return None, None


def train_quantum_model(X_train, y_train, qnn, estimator):
    """
    Train quantum model
    
    Args:
        X_train: Training features
        y_train: Training labels
        qnn: Quantum neural network
        estimator: Qiskit estimator
        
    Returns:
        classifier: Trained quantum classifier
    """
    print("🤖 STEP 6: Train Model (REAL TRAINING)")
    
    if not QISKIT_AVAILABLE or qnn is None or estimator is None:
        print("❌ Cannot train quantum model - missing components")
        return None
    
    try:
        # Create quantum classifier with COBYLA optimizer
        classifier = NeuralNetworkClassifier(
            neural_network=qnn,
            optimizer=COBYLA(maxiter=100),
            callback=lambda w, nfev: print(f"🔄 Iteration {nfev}: Loss = {w:.4f}")
        )
        
        print("🚀 Starting quantum training...")
        classifier.fit(X_train, y_train)
        print("✅ Quantum training completed!")
        
        return classifier
        
    except Exception as e:
        print(f"❌ Quantum training failed: {e}")
        return None


def evaluate_quantum_model(classifier, X_test, y_test):
    """
    Evaluate trained quantum model
    
    Args:
        classifier: Trained quantum classifier
        X_test: Test features
        y_test: Test labels
        
    Returns:
        score: Model accuracy
    """
    print("🧪 STEP 7: Evaluate Model")
    
    if classifier is None:
        print("❌ Cannot evaluate - no trained model")
        return 0.0
    
    try:
        score = classifier.score(X_test, y_test)
        print(f"✅ Quantum Model Accuracy: {score:.4f}")
        
        # Make predictions
        predictions = classifier.predict(X_test)
        print(f"📊 Predictions: {predictions}")
        
        return score
        
    except Exception as e:
        print(f"❌ Quantum evaluation failed: {e}")
        return 0.0


def get_quantum_backend():
    """
    Get quantum backend for execution
    
    Returns:
        backend: Quantum backend or None if unavailable
    """
    print("☁️ STEP 9: Run on REAL Quantum Hardware (Optional)")
    
    if not QISKIT_AVAILABLE:
        print("❌ Qiskit not available for quantum backend")
        return None
    
    try:
        service = QiskitRuntimeService()
        backend = service.least_busy(simulator=False)
        print(f"✅ Using backend: {backend}")
        return backend
        
    except Exception as e:
        print(f"⚠️ Could not connect to quantum hardware: {e}")
        print("🔄 Using simulator instead")
        return None


def create_classical_model():
    """
    Create classical model for hybrid comparison
    
    Returns:
        classical_model: Trained classical model
    """
    print("🤖 STEP 11: Create Classical Model for Hybrid")
    
    try:
        classical_model = RandomForestClassifier(n_estimators=50, random_state=42)
        print("✅ Classical Random Forest model created")
        return classical_model
        
    except Exception as e:
        print(f"❌ Classical model creation failed: {e}")
        return None


def hybrid_predict(quantum_classifier, classical_model, features, scaler):
    """
    Hybrid prediction combining quantum and classical models
    
    Args:
        quantum_classifier: Trained quantum model
        classical_model: Trained classical model
        features: Input features
        scaler: Fitted scaler
        
    Returns:
        result: Dictionary with quantum, classical, and final predictions
    """
    print("🔗 STEP 10: Hybrid Prediction")
    
    # Normalize features
    features_normalized = scaler.transform([features])
    
    # Quantum prediction
    quantum_pred = None
    if quantum_classifier is not None:
        try:
            quantum_pred = int(quantum_classifier.predict(features_normalized)[0])
            print(f"⚛️ Quantum prediction: {quantum_pred}")
        except Exception as e:
            print(f"❌ Quantum prediction failed: {e}")
    
    # Classical prediction
    classical_pred = None
    if classical_model is not None:
        try:
            classical_pred = int(classical_model.predict([features])[0])
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
    """
    Main function demonstrating complete quantum training pipeline
    """
    print("🚀 QUANTUM ADR PREDICTION SYSTEM DEMO")
    print("=" * 50)
    
    # STEP 1: Check dependencies
    print("⚛️ STEP 1: Install Required Packages")
    if QISKIT_AVAILABLE:
        print("✅ Qiskit available")
    else:
        print("❌ Qiskit not available")
        print("Please install with: pip install qiskit qiskit-machine-learning scikit-learn numpy")
        return
    
    # STEP 2: Prepare dataset
    X, y, scaler, feature_names = prepare_adr_dataset()
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"📊 Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # STEP 3-5: Build quantum components
    feature_map, ansatz = build_quantum_circuit(X.shape[1])
    qnn, estimator = create_quantum_neural_network(feature_map, ansatz)
    
    if qnn is None or estimator is None:
        print("❌ Cannot proceed without quantum components")
        return
    
    # STEP 6: Train quantum model
    quantum_classifier = train_quantum_model(X_train, y_train, qnn, estimator)
    
    # STEP 7: Evaluate quantum model
    quantum_score = evaluate_quantum_model(quantum_classifier, X_test, y_test)
    
    # STEP 8: Create classical model for comparison
    classical_model = create_classical_model()
    if classical_model is not None:
        classical_model.fit(X_train, y_train)
        classical_score = classical_model.score(X_test, y_test)
        print(f"🤖 Classical Model Accuracy: {classical_score:.4f}")
    
    # STEP 9: Get quantum backend (optional)
    quantum_backend = get_quantum_backend()
    
    # STEP 10: Demonstrate hybrid prediction
    print("\n🔗 STEP 10: Hybrid Prediction Demo")
    print("-" * 30)
    
    # Test with sample data
    sample_features = [45, 75, 3, 2, 0.7]  # Middle-aged, medium-high risk
    hybrid_result = hybrid_predict(quantum_classifier, classical_model, sample_features, scaler)
    
    print("\n🎉 FINAL RESULTS:")
    print(f"📊 Quantum accuracy: {quantum_score:.4f}")
    if classical_model is not None:
        classical_score = classical_model.score(X_test, y_test)
        print(f"🤖 Classical accuracy: {classical_score:.4f}")
    print(f"🔗 Hybrid prediction for sample: {hybrid_result}")
    
    # Feature importance
    if classical_model is not None:
        print(f"\n📈 Classical Feature Importance:")
        for i, importance in enumerate(classical_model.feature_importances_):
            print(f"  {feature_names[i]}: {importance:.4f}")
    
    print("\n✅ Quantum ADR Prediction Demo Complete!")
    print("🚀 Ready for integration with hybrid system!")


if __name__ == "__main__":
    main()
