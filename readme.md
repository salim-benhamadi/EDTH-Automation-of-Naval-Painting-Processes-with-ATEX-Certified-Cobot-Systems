# Installation Guide

## Prerequisites

### Step 0: Hardware Requirements
- **NVIDIA GPU** (required for ZED SDK and CUDA)
- ZED camera

### Step 1: Install ZED SDK and CUDA

1. **Install CUDA Toolkit**
   - Download and install the appropriate CUDA version from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
   - Verify installation:
     ```bash
     nvcc --version
     ```

2. **Install ZED SDK**
   - Download the ZED SDK from [Stereolabs website](https://www.stereolabs.com/developers/release/)
   - Run the installer and follow the installation wizard
   - Verify installation by checking: `C:\Program Files (x86)\ZED SDK` (Windows) or `/usr/local/zed` (Linux)

---

## Step 2: Create Virtual Environment

```bash
# Create a virtual environment
python -m venv env

# Activate the virtual environment
# On Windows:
env\Scripts\activate

# On Linux/Mac:
source env/bin/activate
```

---

## Step 3: Install ZED Python API

### Clone the Repository

```bash
git clone https://github.com/stereolabs/zed-python-api.git
cd zed-python-api
```

### Navigate to Source Directory

```bash
cd src
```

### Install Dependencies

Choose the appropriate requirements file based on your Python version:

**On Windows:**
```bash
# For Python 3.9+ (recommended)
pip install -r requirements.txt

# OR for older Python versions
pip install -r requirements_legacy.txt
```

**On Linux:**
```bash
# For Python 3.9+ (recommended)
pip3 install -r requirements.txt

# OR for older Python versions
pip3 install -r requirements_legacy.txt
```

### Build and Install

**On Windows:**
```bash
python setup.py build
python setup.py install
python -m pip wheel .
python -m pip install pyzed-*.whl --force-reinstall
```

**On Linux:**
```bash
python3 setup.py build
python3 setup.py install
python3 -m pip wheel .
python3 -m pip install pyzed-*.whl --force-reinstall
```

---

## Step 4: Copy DLL Files (Windows Only)

**⚠️ CRITICAL STEP FOR WINDOWS USERS**

Copy all DLL files from the ZED SDK to your Python environment:

```bash
# Source directory
C:\Program Files (x86)\ZED SDK\bin

# Destination directory (adjust path to your virtual environment)
env\Lib\site-packages\pyzed
```

**To copy:**
1. Open File Explorer and navigate to `C:\Program Files (x86)\ZED SDK\bin`
2. Select all `.dll` files (Ctrl+A)
3. Copy them (Ctrl+C)
4. Navigate to your virtual environment: `<your_project_path>\env\Lib\site-packages\pyzed`
5. Paste the files (Ctrl+V)

---

## Step 5: Install Project Requirements

Navigate back to your project root directory:

```bash
cd ../../..  # or navigate to your project root
```

Create a `requirements.txt` file:

```txt
pyvista>=0.43.0
numpy>=1.24.0
scipy>=1.11.0
gymnasium>=0.29.0
torch>=2.0.0
```

Install the requirements:

**On Windows:**
```bash
pip install -r requirements.txt
```

**On Linux:**
```bash
pip3 install -r requirements.txt
```

---

## Step 6: Verify Installation

Test that everything is installed correctly:

```python
# test_installation.py
import pyzed.sl as sl
import pyvista as pv
import torch
import gymnasium as gym

print("✓ PyZED imported successfully")
print("✓ PyVista imported successfully")
print("✓ PyTorch imported successfully")
print("✓ Gymnasium imported successfully")

# Test ZED camera initialization
zed = sl.Camera()
print(f"✓ ZED SDK version: {sl.get_sdk_version()}")
```

Run the test:
```bash
python test_installation.py
```

---

## Step 7: Run the Project

```bash
# Step 1: Scan and detect
python 1_scan_and_detect.py cup

# Step 2: Filter and isolate
python 2_filter_and_isolate.py cup

# Step 3: Train and test RL agent
python 3_rl_painter.py cup
```

---

## Troubleshooting

### Common Issues

**1. "DLL load failed" error on Windows**
- Make sure you completed Step 4 (copying DLL files)
- Verify the DLL files are in the correct location

**2. CUDA not found**
- Ensure NVIDIA drivers are up to date
- Verify CUDA installation with `nvcc --version`
- Check that your GPU is CUDA-compatible

**3. ZED SDK initialization fails**
- Ensure ZED camera is properly connected
- Run ZED Diagnostic tool from the SDK installation
- Check USB 3.0 connection

**4. Import errors**
- Make sure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

---

## Additional Notes

- The project requires approximately 5-10 GB of disk space
- First-time CUDA setup may require a system restart
- Training the RL agent may take 30+ minutes depending on your GPU