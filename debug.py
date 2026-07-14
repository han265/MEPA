import torch, sys, os
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torch path:", torch.__file__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))