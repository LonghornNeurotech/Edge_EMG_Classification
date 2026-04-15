# MLResourceUse.py
# Standalone resource monitor for the Raspberry Pi Zero 2 W.
# Waits for run_inference.py to create a signal file, then begins
# logging hardware metrics. Stops automatically when inference ends.
#
# USAGE (two terminals):
#   Terminal 1:  python MLResourceUse.py     (will wait for inference)
#   Terminal 2:  python run_inference.py      (triggers the monitor)

import time
import csv
import subprocess
import psutil
from gpiozero import CPUTemperature

SIGNAL_FILE = ".inference_running"   # must match run_inference.py
OUTPUT_CSV = "rpi_profile.csv"
POLL_INTERVAL = 0.2  # seconds between samples while profiling
WAIT_INTERVAL = 0.5  # seconds between checks while waiting

cpu_temp_sensor = CPUTemperature()


def get_cpu_freq_mhz():
    #Read the actual ARM clock frequency via vcgencmd.
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_clock", "arm"]
        ).decode()
        return int(out.split("=")[1]) / 1e6
    except Exception:
        return None


def get_core_voltage():
    # Read the core voltage via vcgencmd.
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_volts", "core"]
        ).decode()
        return float(out.split("=")[1].replace("V", ""))
    except Exception:
        return None


def wait_for_inference():
    # Block until the signal file appears (inference has started).
    import os
    print("=" * 55)
    print("  Resource Monitor — waiting for inference to start...")
    print("  (run 'python run_inference.py' in the other terminal)")
    print("=" * 55)
    while not os.path.exists(SIGNAL_FILE):
        time.sleep(WAIT_INTERVAL)
        
    time.sleep(0.5) # wait half a second for run_inference to finish writing
    try:
        with open(SIGNAL_FILE, "r") as f:
            model_name = f.read().strip()
    except Exception:
        model_name = "unknown_model"
    if not model_name: 
        model_name = "unknown_model"
        
    model_name_clean = model_name.replace(".onnx", "").replace("/", "_")
    print(f"\n>> Signal detected for '{model_name_clean}'! Profiling started.\n")
    return model_name_clean


def inference_is_running():
    """Check if the signal file still exists."""
    import os
    return os.path.exists(SIGNAL_FILE)


def monitor_once():
    clean_name = wait_for_inference()
    output_csv = f"rpi_profile_{clean_name}.csv"

    # Prime psutil so the first real reading isn't zero
    psutil.cpu_percent(interval=None, percpu=True)

    start_time = time.time()
    sample_count = 0
    total_energy_j = 0.0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "elapsed_s",
            "cpu_temp_c",
            "usage_c0",
            "usage_c1",
            "usage_c2",
            "usage_c3",
            "cpu_avg_pct",
            "ram_used_mb",
            "ram_pct",
            "cpu_freq_mhz",
            "core_voltage_v",
            "est_power_w",
            "est_energy_j",
            "cumulative_energy_j",
        ])

        prev_time = time.time()

        while inference_is_running():
            now = time.time()
            dt = now - prev_time  # time since last sample
            elapsed = now - start_time

            # ---- Gather hardware readings ----
            cpu_temp = cpu_temp_sensor.temperature
            cpu_utils = psutil.cpu_percent(interval=None, percpu=True)
            cpu_avg = sum(cpu_utils) / len(cpu_utils)
            ram = psutil.virtual_memory()
            freq_mhz = get_cpu_freq_mhz()
            core_v = get_core_voltage()

            # ---- Estimate power & energy ----
            # P_dyn ≈ α · C · V² · f
            power_w = None
            energy_j = None
            if freq_mhz is not None and core_v is not None:
                alpha = cpu_avg / 100.0
                C = 1e-9  # approximate switching capacitance
                power_w = alpha * C * (core_v ** 2) * (freq_mhz * 1e6)
                energy_j = power_w * dt
                total_energy_j += energy_j

            # ---- Write row ----
            writer.writerow([
                round(elapsed, 4),
                round(cpu_temp, 2),
                round(cpu_utils[0], 2) if len(cpu_utils) > 0 else "",
                round(cpu_utils[1], 2) if len(cpu_utils) > 1 else "",
                round(cpu_utils[2], 2) if len(cpu_utils) > 2 else "",
                round(cpu_utils[3], 2) if len(cpu_utils) > 3 else "",
                round(cpu_avg, 2),
                round(ram.used / (1024 ** 2), 2),
                round(ram.percent, 2),
                round(freq_mhz, 2) if freq_mhz else "",
                round(core_v, 4) if core_v else "",
                round(power_w, 6) if power_w else "",
                round(energy_j, 9) if energy_j else "",
                round(total_energy_j, 6),
            ])
            f.flush()

            # ---- Live terminal output ----
            sample_count += 1
            print(
                f"[{elapsed:6.1f}s] "
                f"Temp: {cpu_temp:5.1f}°C | "
                f"CPU: {cpu_avg:5.1f}% | "
                f"Freq: {freq_mhz:7.1f} MHz | "
                f"Volt: {core_v:.4f} V | "
                f"Power: {power_w:.6f} W | "
                f"RAM: {ram.percent:4.1f}%"
                if freq_mhz and core_v and power_w else
                f"[{elapsed:6.1f}s] "
                f"Temp: {cpu_temp:5.1f}°C | "
                f"CPU: {cpu_avg:5.1f}% | "
                f"RAM: {ram.percent:4.1f}%"
            )

            prev_time = now
            time.sleep(POLL_INTERVAL)

    # ---- Final summary ----
    total_time = time.time() - start_time
    avg_power = total_energy_j / total_time if total_time > 0 else 0

    print(f"\n{'='*55}")
    print(f"  Inference ended. Profiling complete.")
    print(f"{'='*55}")
    print(f"  Duration:           {total_time:.2f} s")
    print(f"  Samples recorded:   {sample_count}")
    print(f"  Est. Total Energy:  {total_energy_j:.6f} J")
    print(f"  Est. Avg Power:     {avg_power:.6f} W")
    print(f"  Data saved to:      {output_csv}")
    print(f"{'='*55}\n")


def main():
    try:
        while True:
            monitor_once()
            print(">> Waiting for next model run... (Ctrl+C to quit)\n")
    except KeyboardInterrupt:
        print("\n>> Resource Monitor exiting.")

if __name__ == "__main__":
    main()