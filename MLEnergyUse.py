#!pip install zeus-ml

import torch
from torch.utils.data import DataLoader as DL, TensorDataset as TData
from zeus.monitor import ZeusMonitor
import argparse
import logging

argparser = argparse.ArgumentParser(description="Measure energy consumption of a PyTorch model using ZeusMonitor.")
argparser.add_argument("--model_path", type=str, help="Path to the PyTorch model file.")
argparser.add_argument("--dataset", type=str, help="Path to the dataset file (PyTorch tensor).")
args = argparser.parse_args()
logging.getLogger("zeus.device.gpu.amd").setLevel(logging.CRITICAL)  # Suppress AMD GPU warnings

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()], approx_instant_energy=True)

    model = torch.jit.load(args.model_path) if args.model_path is not None else torch.load("quantized_model_qat.pth", weights_only=False) # your PyTorch model
    ds = torch.load(args.dataset) if args.dataset is not None else TData(torch.randn((10000, 8, 100)), torch.randint(0, 6, (10000,)))      # your dataset
    dl = DL(ds, batch_size=32, drop_last=True)  # DataLoader for your dataset

    total_time = 0.0
    t0 = 0.0
    t1 = 0.0
    total_energy = 0.0
    test_accuracy = 0.0
    model = model.to(device).eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dl):
            x = x.to(device)
            monitor.begin_window(f"inference{i}")
            y_pred = model(x)
            measurement = monitor.end_window(f"inference{i}")
            total_time += measurement.time
            total_energy += measurement.total_energy
            #test_accuracy += (y_pred == y).float().mean().item()
            

    print("Energy (J):", total_energy)
    print("Average power (W):", total_energy / total_time)
    print("J / inference:", total_energy / (len(dl) * 32))
    print("Average inference time per sample (s):", total_time / (len(dl) * 32))
    #print("Test accuracy:", test_accuracy / len(dl))

except Exception as e:
    print("Error during monitoring:", e)

