import torch
import sys

def print_system_info():
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch built with CUDA: {torch.version.cuda}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"Device count: {torch.cuda.device_count()}")
        current_device = torch.cuda.current_device()
        print(f"Current device ID: {current_device}")
        print(f"Device name: {torch.cuda.get_device_name(current_device)}")
        props = torch.cuda.get_device_properties(current_device)
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("Error: CUDA is not available.")

if __name__ == "__main__":
    print_system_info()