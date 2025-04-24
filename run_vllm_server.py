#!/usr/bin/env python
"""
vLLM Server Runner Script for Stock Market Prediction App

This script provides utilities to run the vLLM server with an OpenAI-compatible API
endpoint that the Django application can connect to.

Prerequisites:
- vLLM installed: pip install vllm
- Suitable model accessible (e.g., Qwen/Qwen2.5-7B-Instruct)
- CUDA-compatible GPU (for better performance)

Usage:
    python run_vllm_server.py [--model MODEL_NAME] [--port PORT] [--gpu-memory-utilization GPU_MEM] [--tensor-parallel-size TP_SIZE]

Example:
    python run_vllm_server.py  # Run with default settings from .env
    python run_vllm_server.py --model meta-llama/Llama-3-8B-Instruct --port 8001  # Override model and port
"""

import argparse
import os
import subprocess
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run_vllm_server(model=None, port=8000, gpu_memory_utilization=0.8, tensor_parallel_size=1):
    """
    Run the vLLM server with the specified parameters
    """
    # Use the model from .env if not specified
    if not model:
        model = os.environ.get('VLLM_MODEL_NAME', 'Qwen/Qwen2.5-7B-Instruct')
    
    # Build the command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", model,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--tensor-parallel-size", str(tensor_parallel_size)
    ]
    
    print(f"Starting vLLM server with model: {model}")
    print(f"Server will be available at: http://localhost:{port}/v1")
    print(f"To stop the server, press Ctrl+C")
    
    try:
        # Execute the command
        process = subprocess.run(cmd)
        return process.returncode
    except KeyboardInterrupt:
        print("\nStopping vLLM server...")
        return 0
    except Exception as e:
        print(f"Error running vLLM server: {e}")
        return 1

def check_vllm_installation():
    """
    Check if vLLM is installed and provide installation instructions if needed
    """
    try:
        import vllm
        print(f"vLLM version {vllm.__version__} is installed.")
        return True
    except ImportError:
        print("vLLM is not installed. Installing vLLM...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "vllm"], check=True)
            print("vLLM has been installed successfully.")
            return True
        except subprocess.CalledProcessError:
            print("\nFailed to install vLLM automatically.")
            print("\nPlease install vLLM manually with one of the following commands:")
            print("\nFor CUDA 11.8:")
            print("pip install vllm")
            print("\nFor CUDA 12.1:")
            print("pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121")
            print("\nFor more information, visit: https://docs.vllm.ai/en/latest/getting_started/installation.html")
            return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the vLLM server for the Stock Market Prediction app")
    parser.add_argument("--model", type=str, help="The model name to use (default: from .env file)")
    parser.add_argument("--port", type=int, default=8000, help="The port to run the server on (default: 8000)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, 
                      help="GPU memory utilization (default: 0.8)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, 
                      help="Tensor parallel size (default: 1)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Check if vLLM is installed
    if not check_vllm_installation():
        sys.exit(1)
    
    # Detect GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Found {gpu_count} GPU(s): {gpu_name}")
            print(f"CUDA Version: {torch.version.cuda}")
            
            # For Intel Arc GPUs, try to detect via other means
            if "Intel" in gpu_name or "Arc" in gpu_name:
                print("Intel GPU detected. Make sure you have Intel Extension for PyTorch installed.")
                print("You may need to set environment variable CUDA_VISIBLE_DEVICES=0")
        else:
            print("No GPU detected. vLLM will run much slower on CPU.")
            user_input = input("Do you want to continue without GPU? (y/n): ")
            if user_input.lower() != 'y':
                sys.exit(0)
    except ImportError:
        print("PyTorch not installed. GPU detection skipped.")
    
    # Run the vLLM server
    sys.exit(run_vllm_server(
        model=args.model,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size
    ))
