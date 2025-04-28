import subprocess
import sys

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")

if __name__ == "__main__":
    install_requirements()
    
    # Import các thư viện sau khi cài đặt
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from vnstock import Vnstock


    print("All libraries imported successfully.")


