import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import time

# 1. Load data
print("Loading dataset for MLP Training...")
data = pd.read_csv("gesture_dataset_cleaned.csv")
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
y = data["label"]
X = data.drop("label", axis=1).select_dtypes(include=['float64', 'int64'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Optimized MLP for Edge (Jetson Nano)
# hidden_layer_sizes=(64, 32): Small enough to be fast, deep enough to learn.
# activation='relu': Standard for neural networks.
# max_iter=500: Ensures it converges.
model = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), 
                          activation='relu', 
                          solver='adam', 
                          max_iter=500,
                          random_state=42))
])

print("Training Multi-Layer Perceptron (Neural Network)...")
start_train = time.time()
model.fit(X_train, y_train)
print(f"MLP Training Time: {time.time() - start_train:.2f}s")

# 3. Benchmark Inference (The Latency Test)
sample = X_test.iloc[0:1]
start_inf = time.time()
for _ in range(100):
    model.predict_proba(sample)
avg_latency = ((time.time() - start_inf) / 100) * 1000

print("-" * 30)
print(f"MLP Model Latency: {avg_latency:.2f}ms") 
print(f"MLP Accuracy: {model.score(X_test, y_test):.4f}")
print("-" * 30)

# 4. Save (Overwrites old model so you can test immediately)
joblib.dump(model, "gesture_mlp_model.pkl", protocol=4) 

print("MLP Model saved successfully!")
