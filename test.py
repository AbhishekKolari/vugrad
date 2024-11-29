# # In Python
# import sys
# import subprocess
# import wmi

# def get_nvidia_driver_version():
#     try:
#         if sys.platform == 'win32':
#             w = wmi.WMI()
#             for driver in w.Win32_VideoController():
#                 if 'nvidia' in driver.Name.lower():
#                     return driver.DriverVersion
#     except:
#         return "Could not detect driver version"

# print(f"NVIDIA Driver Version: {get_nvidia_driver_version()}")

# w = wmi.WMI()
# for gpu in w.Win32_VideoController():
#     print(gpu.Name)

# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA version: {torch.version.cuda}")
# print(f"Is CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# import sys
# import subprocess

# def get_gpu_info():
#     try:
#         if sys.platform == 'win32':
#             import wmi
#             w = wmi.WMI()
#             for gpu in w.Win32_VideoController():
#                 print(f"GPU: {gpu.Name}")
#                 print(f"Driver Version: {gpu.DriverVersion}")
#     except Exception as e:
#         print(f"Error getting GPU info: {e}")

# get_gpu_info()

# import torch

# # Check CUDA availability
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
# else:
#     device = torch.device("cpu")
#     print("Using CPU")

# # Try to create a tensor on GPU
# try:
#     x = torch.rand(5,3).to(device)
#     print("Successfully created tensor on", device)
# except Exception as e:
#     print(f"Error creating tensor: {e}")

import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)