import torch
import cv2
import numpy as np
import onnx
import albumentations
import labelme
import matplotlib
import netron
import tensorboard

print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("ONNX:", onnx.__version__)
print("Albumentations:", albumentations.__version__)
print("Labelme:", labelme.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Netron: OK")
print("TensorBoard: OK")
print("\n所有库安装成功！")