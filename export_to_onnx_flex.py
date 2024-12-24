import os
import torch
import torch.onnx

from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import argparse


"""
Example:
python export_to_onnx_flex.py --encoder vits --image_shape (3, 308, 308)
"""

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
# parser.add_argument('--load_from', type=str, default='./checkpoints/depth_anything_vits14.pth')
# parser.add_argument('--image_shape', type=tuple, default=(3, 308, 308))
parser.add_argument('--image_shape', type=int, default=308)

args = parser.parse_args()

encoder = args.encoder
# load_from = args.load_from
# load_from = './checkpoints/depth_anything_' + encoder + '14_' + str(args.image_shape) + '.pth'
load_from = './checkpoints/depth_anything_' + encoder + '14.pth'
image_shape = (3, args.image_shape, args.image_shape)

# Initializing model
assert encoder in ['vits', 'vitb', 'vitl']
if encoder == 'vits':
    depth_anything = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], localhub='localhub')
elif encoder == 'vitb':
    depth_anything = DPT_DINOv2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768], localhub='localhub')
else:
    depth_anything = DPT_DINOv2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], localhub='localhub')

total_params = sum(param.numel() for param in depth_anything.parameters())
print('Total parameters: {:.2f}M'.format(total_params / 1e6))

# Loading model weight
depth_anything.load_state_dict(torch.load(load_from, map_location='cpu'), strict=True)

depth_anything.eval()

# Define dummy input data
dummy_input = torch.ones(image_shape).unsqueeze(0)

# Provide an example input to the model, this is necessary for exporting to ONNX
example_output = depth_anything(dummy_input)

onnx_path = load_from.split('/')[-1].split('.pth')[0] + '_' + str(args.image_shape) + '.onnx'

# Export the PyTorch model to ONNX format
torch.onnx.export(depth_anything, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"], verbose=True)

print(f"Model exported to {onnx_path}")
