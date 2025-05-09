import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from natsort import natsorted
import copy
import struct
import json
import time
from datetime import datetime

# Set matplotlib to use English to avoid Chinese font issues
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Initialize paths
DATASET_PATH = './dataset'
RESULT_PATH = './results'
MODEL_NAME = 'improved_vgg_gray_multichannel_gesture_model'
QUANTIZED_MODEL_NAME = 'raspberry_pi_quantized_model'

# Create results directory
os.makedirs(RESULT_PATH, exist_ok=True)

# Frame specifications
img_rows, img_cols = 256, 256  # Adjust to 256x256 input size
img_depth = 19  # Number of frames per video
print(f"Frame dimensions: {img_rows}x{img_cols}x{img_depth}")

# Gesture class names
gesture_names = [
    "Gesture_1", "Gesture_2", "Gesture_3", "Gesture_4",
    "Gesture_5", "Gesture_6", "Gesture_7", "Gesture_8"
]

def extract_frames(video_path, width, height, num_frames):
    """Extract grayscale frames from a video"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None
        
        # Get total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Video has no frames: {video_path}")
            return None
            
        # Calculate sampling interval
        if total_frames <= num_frames:
            # Video has fewer frames than required, sample all
            frame_indices = range(total_frames)
        else:
            # Uniform sampling
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        for idx in frame_indices:
            # Set read position
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization to enhance contrast
            gray = cv2.equalizeHist(gray)
            
            frames.append(gray)
        
        cap.release()
        
        # Handle cases with insufficient frames
        if len(frames) < num_frames:
            # If at least one frame exists, duplicate existing frames to fill
            if frames:
                while len(frames) < num_frames:
                    # Cycle through existing frames
                    idx_to_copy = len(frames) % len(frames)
                    frames.append(frames[idx_to_copy].copy())
            else:
                print(f"Could not extract frames: {video_path}")
                return None
    
    except Exception as e:
        print(f"Error processing video: {video_path}")
        print(f"Error: {str(e)}")
        return None
    
    return frames

def load_dataset(dataset_path, max_videos_per_class=3):
    """Load video data from dataset directory"""
    X_data = []
    y_data = []
    
    # Get the first 8 class folders
    class_folders = []
    for i in range(1, 9):  # Classes 1 to 8
        folder_name = f"class_0{i}" if i < 10 else f"class_{i}"
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            class_folders.append((i-1, folder_path))  # Class index starts from 0
    
    if not class_folders:
        print(f"No class folders found in {dataset_path}")
        return np.array([]), np.array([])
    
    # Process each class
    for class_idx, class_path in class_folders:
        print(f"Processing class {class_idx+1}: {os.path.basename(class_path)}")
        
        # Get user folders
        user_folders = [f for f in os.listdir(class_path) 
                        if os.path.isdir(os.path.join(class_path, f))]
        
        if not user_folders:
            print(f"No user folders found in {class_path}")
            continue
        
        # Natural sort the user folders
        user_folders = natsorted(user_folders)
        
        videos_in_class = 0
        
        # Process each user folder
        for user_folder in user_folders:
            user_path = os.path.join(class_path, user_folder)
            
            # Get video files
            video_files = [f for f in os.listdir(user_path) 
                          if f.lower().endswith(('.avi', '.mp4'))]
            
            if not video_files:
                print(f"No video files found in {user_path}")
                continue
            
            # Random selection, limit the number of videos per class
            if len(video_files) > max_videos_per_class - videos_in_class:
                video_files = random.sample(video_files, max_videos_per_class - videos_in_class)
            
            # Process each video
            for video_file in sorted(video_files):
                video_path = os.path.join(user_path, video_file)
                print(f"  Processing: {user_folder}/{video_file}")
                
                frames = extract_frames(video_path, img_rows, img_cols, img_depth)
                
                if frames is not None and len(frames) == img_depth:
                    # Convert to numpy array
                    frames_array = np.array(frames)
                    
                    # Ensure correct frame dimensions
                    if frames_array.shape == (img_depth, img_rows, img_cols):
                        # Add to dataset
                        X_data.append(frames_array)
                        y_data.append(class_idx)
                        
                        videos_in_class += 1
                        if videos_in_class >= max_videos_per_class:
                            break
            
            if videos_in_class >= max_videos_per_class:
                break
        
        print(f"  Total videos for class {class_idx+1}: {videos_in_class}")
    
    return np.array(X_data), np.array(y_data)

class ImprovedGrayMultiChannelTinyVGG(nn.Module):
    """
    Improved Tiny-VGG architecture with residual connections and deeper structure
    """
    def __init__(self, num_input_channels, num_classes):
        super(ImprovedGrayMultiChannelTinyVGG, self).__init__()
        
        # First convolutional block with increased channels and residual connection
        self.conv1_main = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # First block skip connection (projection)
        self.conv1_skip = nn.Sequential(
            nn.Conv2d(num_input_channels, 96, kernel_size=1, stride=2)
        )
        
        # Second convolutional block with increased channels and residual connection
        self.conv2_main = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Second block skip connection (projection)
        self.conv2_skip = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, stride=2)
        )
        
        # Third convolutional block (new addition)
        self.conv3_main = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Improved classifier with two fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Increased dropout ratio for better generalization
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # First block with residual connection
        x1_main = self.conv1_main(x)
        x1_skip = self.conv1_skip(x)
        x1 = x1_main + x1_skip
        
        # Second block with residual connection
        x2_main = self.conv2_main(x1)
        x2_skip = self.conv2_skip(x1)
        x2 = x2_main + x2_skip
        
        # Third block (no residual connection)
        x3 = self.conv3_main(x2)
        
        # Global pooling and classification
        x = self.global_pool(x3)
        x = self.classifier(x)
        return x

def fuse_conv_bn(conv, bn):
    """
    Fuse convolutional layer and batch normalization layer
    
    Parameters:
    conv: nn.Conv2d layer
    bn: nn.BatchNorm2d layer
    
    Returns:
    Fused nn.Conv2d layer
    """
    # Ensure the network is in evaluation mode
    conv.eval()
    bn.eval()
    
    # Get convolution layer parameters
    w_conv = conv.weight.clone().detach()
    if conv.bias is not None:
        b_conv = conv.bias.clone().detach()
    else:
        b_conv = torch.zeros(conv.out_channels, device=w_conv.device)
    
    # Get BN layer parameters
    w_bn = bn.weight.clone().detach()
    b_bn = bn.bias.clone().detach()
    running_mean = bn.running_mean.clone().detach()
    running_var = bn.running_var.clone().detach()
    eps = bn.eps
    
    # Calculate fused weights and biases
    factor = w_bn / torch.sqrt(running_var + eps)
    w_fused = w_conv * factor.reshape(-1, 1, 1, 1)
    b_fused = b_conv * factor + b_bn - running_mean * factor
    
    # Create new convolution layer
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True
    )
    
    # Set fused parameters
    fused_conv.weight.data = w_fused
    fused_conv.bias.data = b_fused
    
    return fused_conv

# Move ConvReLU class to module level
class ConvReLU(nn.Conv2d):
    def forward(self, x):
        return torch.relu(super().forward(x))

def fuse_conv_bn_relu(conv, bn, relu):
    """
    Fuse convolutional, batch normalization, and ReLU layers
    
    Parameters:
    conv: nn.Conv2d layer
    bn: nn.BatchNorm2d layer
    relu: nn.ReLU layer
    
    Returns:
    Fused layer
    """
    # First fuse conv and BN
    fused_conv = fuse_conv_bn(conv, bn)
    
    # Create a convolution layer with fused ReLU
    fused_conv_relu = ConvReLU(
        fused_conv.in_channels,
        fused_conv.out_channels,
        fused_conv.kernel_size,
        fused_conv.stride,
        fused_conv.padding,
        fused_conv.dilation,
        fused_conv.groups,
        bias=True
    )
    
    # Copy parameters from the fused convolution layer
    fused_conv_relu.weight.data = fused_conv.weight.data
    fused_conv_relu.bias.data = fused_conv.bias.data
    
    return fused_conv_relu

def fuse_model_layers(model):
    """
    Fuse convolution, BN, and ReLU layers in the model
    
    Parameters:
    model: PyTorch model
    
    Returns:
    Fused model
    """
    # Create a model copy
    fused_model = copy.deepcopy(model)
    fused_model.eval()  # Set to evaluation mode
    
    # Find and fuse Conv+BN+ReLU sequences in conv1_main
    if hasattr(fused_model, 'conv1_main') and isinstance(fused_model.conv1_main, nn.Sequential):
        seq = fused_model.conv1_main
        
        # Process first Conv+BN+ReLU group
        if len(seq) >= 3 and isinstance(seq[0], nn.Conv2d) and isinstance(seq[1], nn.BatchNorm2d) and isinstance(seq[2], nn.ReLU):
            fused_conv_relu = fuse_conv_bn_relu(seq[0], seq[1], seq[2])
            new_seq = nn.Sequential()
            new_seq.add_module('0', fused_conv_relu)
            
            # Add remaining layers
            for i in range(3, len(seq)):
                new_seq.add_module(str(i-2), seq[i])
            
            # Also fuse the second Conv+BN+ReLU group if it exists
            if len(new_seq) >= 4 and isinstance(new_seq[1], nn.Conv2d) and isinstance(new_seq[2], nn.BatchNorm2d) and isinstance(new_seq[3], nn.ReLU):
                fused_conv_relu2 = fuse_conv_bn_relu(new_seq[1], new_seq[2], new_seq[3])
                final_seq = nn.Sequential()
                final_seq.add_module('0', new_seq[0])
                final_seq.add_module('1', fused_conv_relu2)
                
                # Add remaining layers
                for i in range(4, len(new_seq)):
                    final_seq.add_module(str(i-2), new_seq[i])
                
                fused_model.conv1_main = final_seq
            else:
                fused_model.conv1_main = new_seq
    
    # Perform the same fusion for conv2_main and conv3_main
    # Additional layer fusion processing can be added here as needed
    
    return fused_model

def calculate_flops_and_params(model, input_tensor):
    """
    Calculate model FLOPs and parameter count
    
    Parameters:
    model: PyTorch model
    input_tensor: Example input tensor
    
    Returns:
    flops, params: Floating point operations and parameter count
    """
    from collections import Counter
    import numpy as np

    # Record hooks
    flops_count = 0
    params_count = 0
    
    # Hook functions
    def conv2d_hook(module, input, output):
        nonlocal flops_count, params_count
        batch_size = input[0].size(0)
        input_channels = module.in_channels
        output_channels = module.out_channels
        kernel_h, kernel_w = module.kernel_size
        output_h, output_w = output.size(2), output.size(3)
        
        # Each output element requires kernel_h * kernel_w * input_channels multiplications and additions
        flops_per_element = 2 * kernel_h * kernel_w * input_channels  # Count both multiplications and additions
        flops_per_batch = flops_per_element * output_h * output_w * output_channels
        flops_count += flops_per_batch * batch_size
        
        # Parameter count: kernels + biases
        params_count += module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
    
    def linear_hook(module, input, output):
        nonlocal flops_count, params_count
        batch_size = input[0].size(0)
        flops_count += 2 * batch_size * module.in_features * module.out_features  # Count both multiplications and additions
        params_count += module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv2d_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    
    # Forward pass
    model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return flops_count, params_count

def quantize_for_raspberry_pi(model, target_bits=8, export_path=None):
    """
    Quantize model weights to lower bit-width format, suitable for Raspberry Pi deployment
    
    Parameters:
    model (nn.Module): Pretrained PyTorch model
    target_bits (int): Target bit width for weights
    export_path (str): Export path
    
    Returns:
    Quantized model, quantization parameters, and statistics
    """
    print(f"Starting model quantization for Raspberry Pi... (Target: {target_bits}-bit)")
    
    # For collecting quantization parameters
    quant_params = {}
    
    # Create a copy of the model for quantization
    quantized_model = copy.deepcopy(model)
    quantized_model.eval()
    
    # Calculate quantization min and max values
    if target_bits == 8:
        quant_min, quant_max = -128, 127
    elif target_bits == 4:
        quant_min, quant_max = -8, 7
    else:
        raise ValueError(f"Bit width not supported for Raspberry Pi deployment: {target_bits}")
    
    # Quantization error statistics
    total_params = 0
    total_error = 0
    model_size_bytes = 0
    
    # Quantize each parameter
    for name, param in quantized_model.named_parameters():
        if 'weight' in name or 'bias' in name:  # Quantize weights and biases for Raspberry Pi
            weight_tensor = param.data
            total_params += weight_tensor.numel()
            
            # Calculate min and max values for each layer for quantization
            tensor_min = torch.min(weight_tensor)
            tensor_max = torch.max(weight_tensor)
            
            # Use symmetric quantization
            max_abs = max(abs(tensor_min.item()), abs(tensor_max.item()))
            
            # Calculate scaling factor - non-power of 2, use floating-point scale for higher precision
            scale = max_abs / quant_max
            
            # Avoid division by zero
            if scale == 0:
                scale = 1e-10
                
            # Quantize to target bit width range
            weight_quantized = torch.round(weight_tensor / scale).clamp(quant_min, quant_max)
            
            # Calculate and accumulate quantization error
            error = torch.mean(torch.abs(weight_tensor - (weight_quantized * scale)))
            total_error += error.item() * weight_tensor.numel()
            
            # Update model parameters to quantized values (for evaluation)
            param.data = weight_quantized * scale
            
            # Calculate quantized parameter size (bytes)
            param_size_bytes = weight_tensor.numel() * (target_bits / 8)
            model_size_bytes += param_size_bytes
            
            # Save quantization parameters
            quant_params[name] = {
                'scale': scale,
                'min': tensor_min.item(),
                'max': tensor_max.item(),
                'quantized_min': weight_quantized.min().item(),
                'quantized_max': weight_quantized.max().item(),
                'bits': target_bits,
                'error': error.item(),
                'size_bytes': param_size_bytes
            }
    
    # Calculate average quantization error
    avg_error = total_error / total_params
    
    # Calculate model size
    model_size_kb = model_size_bytes / 1024
    model_size_mb = model_size_kb / 1024
    
    # Print quantization statistics
    print(f"Quantization complete! Average quantization error: {avg_error:.6f}")
    print(f"Quantized model size: {model_size_bytes:.0f} bytes ({model_size_kb:.2f} KB, {model_size_mb:.4f} MB)")
    
    quant_stats = {
        'avg_error': avg_error,
        'target_bits': target_bits,
        'total_params': total_params,
        'model_size_bytes': model_size_bytes,
        'model_size_kb': model_size_kb,
        'model_size_mb': model_size_mb
    }
    
    return quantized_model, quant_params, quant_stats

def optimize_for_raspberry_pi(model, input_shape=(1, 19, 256, 256)):
    """
    Optimize model to run faster on Raspberry Pi 5
    
    Parameters:
    model (nn.Module): PyTorch model
    input_shape (tuple): Input tensor shape
    
    Returns:
    Optimized model
    """
    print("Optimizing model for improved performance on Raspberry Pi 5...")
    
    # 1. Layer fusion - fuse Conv, BN, and ReLU layers
    fused_model = fuse_model_layers(model)
    
    # 2. Generate performance report
    with torch.no_grad():
        dummy_input = torch.randn(*input_shape)
        flops, params = calculate_flops_and_params(fused_model, dummy_input)
        print(f"  Model parameters: {params/1e6:.2f}M")
        print(f"  Estimated FLOPs: {flops/1e9:.2f}G")
        
    return fused_model


def quantize_model_with_gptq(model, calibration_loader, target_bits=8, block_size=128, export_path=None):
    """
    Quantize model using GPTQ (Gradient-based Post-Training Quantization) method
    
    Parameters:
    model (nn.Module): Pretrained PyTorch model
    calibration_loader: Data loader for calibration
    target_bits (int): Target bit width for weights
    block_size (int): Block size
    export_path (str): Export path
    
    Returns:
    Quantized model, quantization parameters, and statistics
    """
    """Fixed version of GPTQ quantization function"""
    print(f"Starting model quantization using GPTQ method... (Target: {target_bits}-bit)")
    
    # Ensure the model is in evaluation mode
    model.eval()
    quantized_model = copy.deepcopy(model)
    
    # Initialize quantization statistics
    quant_params = {}
    total_params = 0
    total_error = 0
    model_size_bytes = 0
    
    # Set quantization range
    if target_bits == 8:
        quant_min, quant_max = -128, 127
    elif target_bits == 4:
        quant_min, quant_max = -8, 7
    else:
        raise ValueError(f"Unsupported bit width: {target_bits}")
    
    # Quantize each parameter module in the model
    print("Performing GPTQ quantization...")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Get layer weights
            weight = module.weight.data.clone()
            total_params += weight.numel()
            
            # Simplified quantization method - without using Hessian information
            orig_shape = weight.shape
            weight_flat = weight.view(-1)
            num_elements = weight_flat.numel()
            
            # Calculate global scaling factor
            max_val = torch.max(torch.abs(weight_flat))
            scale = max_val / quant_max if max_val > 0 else 1e-10
            
            # Perform quantization
            weight_quantized = torch.round(weight_flat / scale).clamp(quant_min, quant_max)
            weight_dequantized = weight_quantized * scale
            
            # Calculate quantization error
            error = torch.mean(torch.abs(weight_flat - weight_dequantized))
            total_error += error.item() * num_elements
            
            # Restore original shape
            weight_dequantized = weight_dequantized.view(orig_shape)
            
            # Update model parameters
            module_in_quantized_model = dict([*quantized_model.named_modules()])[name]
            module_in_quantized_model.weight.data = weight_dequantized
            
            # Calculate quantized parameter size
            param_size_bytes = weight.numel() * (target_bits / 8)
            model_size_bytes += param_size_bytes
            
            # Record quantization parameters
            quant_params[name + '.weight'] = {
                'bits': target_bits,
                'error': error.item(),
                'scale': scale.item(),
                'size_bytes': param_size_bytes
            }
            
            print(f"  Quantized {name}.weight: Shape {weight.shape}, Error: {error.item():.6f}")
    
    # Calculate average quantization error
    avg_error = total_error / total_params if total_params > 0 else 0
    
    # Calculate model size
    model_size_kb = model_size_bytes / 1024
    model_size_mb = model_size_kb / 1024
    
    # Print quantization statistics
    print(f"Quantization complete! Average quantization error: {avg_error:.6f}")
    print(f"Quantized model size: {model_size_bytes:.0f} bytes ({model_size_kb:.2f} KB, {model_size_mb:.4f} MB)")
    
    quant_stats = {
        'avg_error': avg_error,
        'target_bits': target_bits,
        'total_params': total_params,
        'model_size_bytes': model_size_bytes,
        'model_size_kb': model_size_kb,
        'model_size_mb': model_size_mb,
        'quantization_method': 'GPTQ-simplified'
    }
    
    return quantized_model, quant_params, quant_stats
    
def prepare_raspberry_pi_deployment_files(model, export_path, model_name):
    """
    Prepare files needed for deployment on Raspberry Pi 5:
      1. Create deployment directory
      2. Save complete PyTorch model
      3. Generate inference.py script
      4. Generate install_dependencies.sh script
      5. Generate README.md deployment instructions
    """
    print("Preparing Raspberry Pi deployment files...")

    # 1. Create deployment directory
    deploy_dir = os.path.join(export_path, f"{model_name}_raspberry_pi")
    os.makedirs(deploy_dir, exist_ok=True)

    # Ensure the model is in evaluation mode
    model.eval()

    # 2. Save complete PyTorch model (including architecture)
    torch_full_model_path = os.path.join(deploy_dir, f"{model_name}_full.pt")
    torch.save(model, torch_full_model_path)

    # 3. Generate Python inference script inference.py
    inference_script = """
import numpy as np
import torch
import cv2
import time
import json
from pathlib import Path

# Load model
def load_model(model_path):
    model = torch.jit.load(model_path) if model_path.endswith('.pt') else torch.load(model_path)
    model.eval()
    return model

def extract_frames(video_path, width=256, height=256, num_frames=19):
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int) if total_frames > num_frames else range(total_frames)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        frames.append(gray)
    cap.release()
    # Fill in missing frames
    while len(frames) < num_frames and frames:
        frames.append(frames[len(frames) % len(frames)].copy())
    return frames

def preprocess(frames):
    arr = np.array(frames, dtype=np.float32) / 255.0
    arr = arr.reshape(1, arr.shape[0], arr.shape[1], arr.shape[2])  # NCHW
    return torch.from_numpy(arr)

def predict(model, input_tensor):
    start = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    end = time.time()
    probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
    return probs, (end - start) * 1000

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python inference.py <model_path> <video_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    video_path = sys.argv[2]

    model = load_model(model_path)
    frames = extract_frames(video_path)
    input_tensor = preprocess(frames)
    probs, latency_ms = predict(model, input_tensor)
    pred = np.argmax(probs[0])
    confidence = probs[0][pred]
    gesture_names = [
        "Gesture_1","Gesture_2","Gesture_3","Gesture_4",
        "Gesture_5","Gesture_6","Gesture_7","Gesture_8"
    ]
    print(f"Predicted gesture: {gesture_names[pred]}, Confidence: {confidence:.4f}, Inference time: {latency_ms:.2f} ms")
"""
    with open(os.path.join(deploy_dir, "inference.py"), 'w', encoding='utf-8') as f:
        f.write(inference_script)

# 4. Generate installation dependencies script install_dependencies.sh
    install_script = """#!/bin/bash
echo "Installing dependencies..."
pip install --no-cache-dir numpy opencv-python torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
echo "Installing additional dependencies..."
pip install --no-cache-dir matplotlib pandas scikit-learn
echo "Dependencies installation complete!"
"""
    install_path = os.path.join(deploy_dir, "install_dependencies.sh")
    with open(install_path, 'w', encoding='utf-8') as f:
        f.write(install_script)
    os.chmod(install_path, 0o755)

    # 5. Generate README.md
    readme = f"""# Raspberry Pi 5 Gesture Recognition Model Deployment

## Files
- `{model_name}_full.pt`: PyTorch native model
- `inference.py`: Inference script
- `install_dependencies.sh`: Dependencies installation script

## Deployment Steps
1. Copy the entire `{model_name}_raspberry_pi` folder to your Raspberry Pi 5
2. Run `./install_dependencies.sh` to install dependencies
3. Run `python inference.py {model_name}_full.pt <video_path>`
4. Check the console output for prediction results and inference time

## Notes
- Input frames: {img_depth}, Resolution: {img_rows}×{img_cols}
- Quantization bit width: 8-bit
- Recommended system: Raspberry Pi OS Bullseye+, Python 3.7+
"""
    with open(os.path.join(deploy_dir, "README.md"), 'w', encoding='utf-8') as f:
        f.write(readme)


def test_quantized_model_inference(model, quantized_model, val_loader, batch_size=4):
    """
    Test the inference performance and accuracy of the quantized model
    
    Parameters:
    model: Original model
    quantized_model: Quantized model
    val_loader: Validation data loader
    batch_size: Batch size
    
    Returns:
    Test results dictionary
    """
    print("\n======== Quantized Model Inference Test ========")
    device = torch.device("cpu")
    model = model.to(device)
    quantized_model = quantized_model.to(device)
    
    model.eval()
    quantized_model.eval()
    
    # Test accuracy
    original_correct = 0
    quantized_correct = 0
    total = 0
    
    # Test inference time
    original_inference_times = []
    quantized_inference_times = []
    
    # Confidence statistics
    original_confidences = []
    quantized_confidences = []
    
    # Quantization error
    output_errors = []
    
    print(f"{'Sample ID':^10}{'True Class':^10}{'Orig Pred':^12}{'Quant Pred':^12}{'Orig Conf':^12}{'Quant Conf':^12}{'Orig Time(ms)':^15}{'Quant Time(ms)':^15}{'Result':^8}")
    print("-" * 100)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Test each sample individually
            for i in range(inputs.size(0)):
                sample_input = inputs[i:i+1]  # Get single sample
                sample_target = targets[i:i+1]
                
                # Test original model inference time
                start_time = time.time()
                original_output = model(sample_input)
                end_time = time.time()
                original_time = (end_time - start_time) * 1000  # milliseconds
                original_inference_times.append(original_time)
                
                # Test quantized model inference time
                start_time = time.time()
                quantized_output = quantized_model(sample_input)
                end_time = time.time()
                quantized_time = (end_time - start_time) * 1000  # milliseconds
                quantized_inference_times.append(quantized_time)
                
                # Calculate output error
                output_error = torch.mean(torch.abs(original_output - quantized_output)).item()
                output_errors.append(output_error)
                
                # Get prediction results
                original_probs = torch.nn.functional.softmax(original_output, dim=1)
                quantized_probs = torch.nn.functional.softmax(quantized_output, dim=1)
                
                _, original_pred = original_output.max(1)
                _, quantized_pred = quantized_output.max(1)
                
                # Get confidence
                original_confidence = original_probs[0][original_pred.item()].item()
                quantized_confidence = quantized_probs[0][quantized_pred.item()].item()
                
                original_confidences.append(original_confidence)
                quantized_confidences.append(quantized_confidence)
                
                # Count accuracy
                true_class = sample_target.argmax(dim=1).item()
                
                if original_pred.item() == true_class:
                    original_correct += 1
                
                if quantized_pred.item() == true_class:
                    quantized_correct += 1
                
                total += 1
                
                # Print detailed results
                result = "Match" if original_pred.item() == quantized_pred.item() else "Mismatch"
                print(f"{batch_idx*batch_size+i:^10}{true_class:^10}{original_pred.item():^12}"
                      f"{quantized_pred.item():^12}{original_confidence:^12.4f}{quantized_confidence:^12.4f}"
                      f"{original_time:^15.2f}{quantized_time:^15.2f}{result:^8}")
                
                # Only test the first 50 samples
                if total >= 50:
                    break
            
            if total >= 50:
                break
    
    original_acc = 100. * original_correct / total
    quantized_acc = 100. * quantized_correct / total
    
    # Calculate average inference time and speedup
    avg_original_time = sum(original_inference_times) / len(original_inference_times)
    avg_quantized_time = sum(quantized_inference_times) / len(quantized_inference_times)
    speedup = avg_original_time / avg_quantized_time
    
    # Calculate average quantization error
    avg_output_error = sum(output_errors) / len(output_errors)
    
    # Calculate average confidence
    avg_original_confidence = sum(original_confidences) / len(original_confidences)
    avg_quantized_confidence = sum(quantized_confidences) / len(quantized_confidences)
    
    # Calculate prediction consistency
    prediction_match_count = sum(1 for i in range(total) 
                               if original_confidences[i] > 0.5 and quantized_confidences[i] > 0.5)
    prediction_consistency = 100. * prediction_match_count / total
    
    # Print statistics
    print("\n======== Inference Test Statistics ========")
    print(f"1. Test samples: {total}")
    print(f"2. Original model accuracy: {original_acc:.2f}%")
    print(f"3. Quantized model accuracy: {quantized_acc:.2f}% (change: {quantized_acc - original_acc:.2f}%)")
    print(f"4. Original model average inference time: {avg_original_time:.2f} ms")
    print(f"5. Quantized model average inference time: {avg_quantized_time:.2f} ms")
    print(f"6. Inference speedup: {speedup:.2f}x")
    print(f"7. Average output error: {avg_output_error:.6f}")
    print(f"8. Original model average confidence: {avg_original_confidence:.4f}")
    print(f"9. Quantized model average confidence: {avg_quantized_confidence:.4f}")
    print(f"10. Prediction consistency: {prediction_consistency:.2f}%")
    
    result_stats = {
        'total_samples': total,
        'original_accuracy': original_acc,
        'quantized_accuracy': quantized_acc,
        'accuracy_change': quantized_acc - original_acc,
        'avg_original_time': avg_original_time,
        'avg_quantized_time': avg_quantized_time,
        'speedup': speedup,
        'avg_output_error': avg_output_error,
        'avg_original_confidence': avg_original_confidence,
        'avg_quantized_confidence': avg_quantized_confidence,
        'prediction_consistency': prediction_consistency
    }
    
    return result_stats

# Enhanced data augmentation function
def apply_augmentations(X_data, y_data, Y_data):
    """
    Memory-efficient data augmentation method - only augments a portion of data at a time
    """
    print("Applying memory-efficient data augmentation...")
    
    # Determine the number of samples to augment - only augment half of the original data
    num_samples = X_data.shape[0]
    num_to_augment = num_samples // 2
    
    # Randomly select samples to augment
    indices = np.random.choice(num_samples, num_to_augment, replace=False)
    
    # Prepare lists for augmented data
    augmented_samples = []
    augmented_labels = []
    
    # Add all original data
    augmented_samples.append(X_data)
    augmented_labels.append(Y_data)
    
    # Apply noise augmentation to selected samples
    X_selected = X_data[indices]
    Y_selected = Y_data[indices]
    
    # Noise augmentation
    noise_factor = 0.05
    X_noisy = X_selected + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=X_selected.shape)
    X_noisy = np.clip(X_noisy, 0., 1.)
    augmented_samples.append(X_noisy)
    augmented_labels.append(Y_selected)
    
    # Combine augmented data
    X_combined = np.concatenate(augmented_samples, axis=0)
    Y_combined = np.concatenate(augmented_labels, axis=0)
    
    print(f"Dataset size after augmentation: {X_combined.shape[0]} samples")
    
    return X_combined, Y_combined

# Training function with early stopping
def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, scheduler, nb_epoch, patience=10):
    """
    Train the model with early stopping
    
    Parameters:
    model: Model
    train_loader: Training data loader
    val_loader: Validation data loader
    criterion: Loss function
    optimizer: Optimizer
    scheduler: Learning rate scheduler
    nb_epoch: Maximum training epochs
    patience: Early stopping patience value
    
    Returns:
    model, train_losses, val_losses, train_accuracies, val_accuracies
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_acc = 0.0
    best_model_state = None
    no_improve_epochs = 0
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(nb_epoch):
        # Training loop
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.argmax(dim=1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.argmax(dim=1)).sum().item()
            
            # Print progress
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{nb_epoch}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets.argmax(dim=1))
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets.argmax(dim=1)).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Print results
        print(f"Epoch [{epoch+1}/{nb_epoch}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            
        if no_improve_epochs >= patience:
            print(f"Early stopping - Epoch {epoch+1}, validation accuracy has not improved for {patience} epochs")
            break
    
    # Load best model state
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, train_accuracies, val_accuracies

def main():
    # Set random seed
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    # Set number of classes
    nb_classes = 8
    
    # Load dataset
    print("Loading dataset...")
    X_data, y_data = load_dataset(DATASET_PATH, max_videos_per_class=3)
    
    num_samples = len(X_data)
    if num_samples == 0:
        print("No valid samples found, please check the dataset path.")
        return
    
    print(f"Total samples loaded: {num_samples}")
    print(f"X_data shape: {X_data.shape}")
    print(f"y_data shape: {y_data.shape}")
    
    # Convert to classification labels
    Y_data = np.eye(nb_classes)[y_data]
    
    # Data normalization
    X_data = X_data.astype('float32')
    X_data /= 255.0
    
    # Apply enhanced data augmentation
    X_combined, Y_combined = apply_augmentations(X_data, y_data, Y_data)
    
    # Split into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_combined, Y_combined, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    
    batch_size = 8  # Appropriate batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model or load pretrained model
    model_path = os.path.join(RESULT_PATH, MODEL_NAME + '.pth')
    
    if os.path.exists(model_path):
        print(f"Loading pretrained model: {model_path}")
        model = ImprovedGrayMultiChannelTinyVGG(img_depth, nb_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print("Creating new model and starting training...")
        model = ImprovedGrayMultiChannelTinyVGG(img_depth, nb_classes)
        
        # Set batch normalization layer parameters
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = 0.1
                m.eps = 1e-5
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Use cosine annealing learning rate schedule
        nb_epoch = 60
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=nb_epoch, eta_min=0.00001
        )
        
        # Use early stopping to train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Train model
        print("Training model...")
        model, train_losses, val_losses, train_accuracies, val_accuracies = train_with_early_stopping(  
            model, train_loader, val_loader, criterion, optimizer, 
            scheduler, nb_epoch, patience=10
        )
        
        # Save trained model
        torch.save(model.state_dict(), os.path.join(RESULT_PATH, MODEL_NAME + '.pth'))
        print('Model successfully saved')
        
        # Plot training curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Acc')
        plt.plot(val_accuracies, label='Val Acc')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_PATH, f"{MODEL_NAME}_training_curves.png"))
        plt.close()
    
    # Ensure model is on CPU for further processing
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    # Validate original model performance
    print("\nEvaluating original model performance...")
    original_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            original_correct += predicted.eq(targets.argmax(dim=1)).sum().item()
    
    original_acc = 100. * original_correct / total
    print(f"Original model accuracy: {original_acc:.2f}%")
    
    # Optimize model - apply layer fusion
    print("\nStarting model optimization...")
    optimized_model = optimize_for_raspberry_pi(model)
    
    # Validate optimized model performance
    print("\nEvaluating optimized model performance...")
    optimized_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = optimized_model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            optimized_correct += predicted.eq(targets.argmax(dim=1)).sum().item()
    
    optimized_acc = 100. * optimized_correct / total
    print(f"Optimized model accuracy: {optimized_acc:.2f}%")
    print(f"Accuracy change after optimization: {optimized_acc - original_acc:.2f}%")
    
    # Use GPTQ to quantize the model
    print("\nStarting model quantization using GPTQ...")
    quantized_model, quant_params, quant_stats = quantize_model_with_gptq(
    optimized_model, val_loader, target_bits=8, block_size=128)
    
    # Validate quantized model performance
    quantized_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = quantized_model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            quantized_correct += predicted.eq(targets.argmax(dim=1)).sum().item()
    
    quantized_acc = 100. * quantized_correct / total
    print(f"Quantized model accuracy: {quantized_acc:.2f}%")
    print(f"Accuracy change after quantization: {quantized_acc - optimized_acc:.2f}%")
    print(f"Total accuracy change (original→quantized): {quantized_acc - original_acc:.2f}%")
    
    # Perform detailed quantized model inference test
    inference_results = test_quantized_model_inference(
        model, quantized_model, val_loader, batch_size=batch_size)
    
    # Prepare Raspberry Pi deployment files
    prepare_raspberry_pi_deployment_files(
        quantized_model, RESULT_PATH, 'improved_gesture_recognition')
    



if __name__ == "__main__":
    main()