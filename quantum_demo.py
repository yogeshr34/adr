"""
Simple Quantum ADR Demo
Working quantum training with Qiskit
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Try quantum imports
try:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.algorithms.classifiers import VQC
    QISKIT_AVAILABLE = True
    print("✅ Qiskit available")
except ImportError:
    QISKIT_AVAILABLE = False
    print("❌ Qiskit not available")

# Dataset
X = np.array([[25,50,1,0],[60,100,1,1],[30,20,0,1],[50,80,1,1]])
y = np.array([0,1,0,1])

# Normalize
scaler = MinMaxScaler(feature_range=(0, np.pi))
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Data: Train {X_train.shape}, Test {X_test.shape}")

# Classical model
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
print(f"Classical accuracy: {rf_score:.4f}")

# Quantum model (if available)
if QISKIT_AVAILABLE:
    try:
        vqc = VQC(
            feature_map=ZZFeatureMap(feature_dimension=4, reps=2),
            ansatz=RealAmplitudes(num_qubits=4, reps=2),
            max_iter=50
        )
        vqc.fit(X_train, y_train)
        quantum_score = vqc.score(X_test, y_test)
        print(f"Quantum accuracy: {quantum_score:.4f}")
        
        # Hybrid prediction
        def hybrid_predict(features):
            features_norm = scaler.transform([features])
            q_pred = vqc.predict(features_norm)[0]
            c_pred = rf.predict([features])[0]
            final = (q_pred + c_pred) / 2
            return int(final > 0.5)
        
        # Test hybrid
        test_features = [45, 75, 1, 1]
        result = hybrid_predict(test_features)
        print(f"Hybrid prediction: {result}")
        
    except Exception as e:
        print(f"Quantum failed: {e}")

print("✅ Demo complete!")
