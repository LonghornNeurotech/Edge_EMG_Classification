import onnxruntime as ort
import numpy as np
import time
import os

SIGNAL_FILE = ".inference_running"

X_test = np.load('X_test.npy').astype(np.float32)
y_test = np.load('y_test.npy')
print(f"Loaded {len(X_test)} samples.\n")

def evaluate_model(model_path, X, y):
    print(f"Evaluating {model_path}")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Warmup
    _ = session.run(None, {input_name: X[0:1]})
    
    start_time = time.time()
    predictions = []
    for i in range(len(X)):
        pred = session.run(None, {input_name: X[i:i+1]})[0]
        predictions.append(np.argmax(pred, axis=1)[0])
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency_ms = (total_time / len(X)) * 1000
    accuracy = np.mean(np.array(predictions) == y)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Avg Latency: {avg_latency_ms:.2f} ms")
    
    # snapshot of predictions
    print("--- Snapshot of First 15 Predictions ---")
    print(f"Predicted: {predictions[:15]}")
    print(f"Actual:    {list(y[:15])}\n")

def run_monitored_inference(model_path):
    # Signal the monitor by writing the model's exact name into the file
    with open(SIGNAL_FILE, "w") as f:
        f.write(model_path)
    print(f">> Signal sent for {model_path}. Inference starting...\n")

    try:
        evaluate_model(model_path, X_test, y_test)
    finally:
        # Delete signal file to tell monitor this specific model finished
        if os.path.exists(SIGNAL_FILE):
            os.remove(SIGNAL_FILE)
        print(f">> Signal removed. {model_path} benchmark complete.\n")

# 1. Run Quantized (8-bit) First
run_monitored_inference('emg_mlp_model_quantized.onnx')

print(">> Waiting 15 seconds for Raspberry Pi CPU to cool down...\n")
time.sleep(15)

# 2. Run Regular (32-bit floating) Second
run_monitored_inference('emg_mlp_model.onnx')

print(">> All Benchmarks Complete!")
