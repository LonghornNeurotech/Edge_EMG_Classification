import onnxruntime as ort
import numpy as np
import time

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
    print(f"Avg Latency: {avg_latency_ms:.2f} ms\n")

evaluate_model('emg_mlp_model.onnx', X_test, y_test)
evaluate_model('emg_mlp_model_quantized.onnx', X_test, y_test)
