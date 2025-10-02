import torch
import sys

print("=" * 50)
print("PyTorch CUDA Setup Verification")
print("=" * 50)

# Basic info
print(f"\nPython version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\nCUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"  Memory cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    
    # Test tensor operations
    print("\n" + "=" * 50)
    print("Testing CUDA Operations")
    print("=" * 50)
    
    try:
        # Create tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Perform operation
        z = torch.matmul(x, y)
        
        print("✓ Successfully created tensors on GPU")
        print("✓ Successfully performed matrix multiplication on GPU")
        print(f"  Result shape: {z.shape}")
        print(f"  Result device: {z.device}")
        
    except Exception as e:
        print(f"✗ Error during GPU operations: {e}")
else:
    print("\n⚠️  CUDA is not available!")
    print("Possible reasons:")
    print("  1. PyTorch was installed without CUDA support")
    print("  2. CUDA drivers are not properly installed")
    print("  3. GPU is not detected")

print("\n" + "=" * 50)