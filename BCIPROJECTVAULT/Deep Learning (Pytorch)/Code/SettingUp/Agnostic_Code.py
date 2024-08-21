import torch;
import argparse;

# Check for GPU access with pytorch
print(torch.cuda.is_available());

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu";

# Count numbers of devices
print(torch.cuda.device_count());

# +==== The code below is from the https://pytorch.org/docs/stable/notes/cuda.html#best-practices website for best practices ====+

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

