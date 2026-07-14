import os
import sys
import torch
from safetensors.torch import load_file

REPO_DIR = "../dinov3"
SRC = "../pretrained_models/model.safetensors"
DST = "../pretrained_models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
MODEL_NAME = "dinov3_vitb16"

sys.path.insert(0, REPO_DIR)

print("loading HF safetensors:", SRC)
hf = load_file(SRC, device="cpu")
print("HF keys:", len(hf))
print("first 20 HF keys:")
for k in list(hf.keys())[:20]:
  print(" ", k)

print()
print("building empty DINOv3 repo model to get target state_dict structure...")
model = torch.hub.load(
  REPO_DIR,
  MODEL_NAME,
  source="local",
  pretrained=False,
)

target = model.state_dict()
converted = {}

def adapt_shape(x, dst_shape, src_key, dst_key):
  """
  Adapt harmless HF <-> DINOv3 repo shape differences.
  Common cases:
    HF token:   (1, 1, C)
    repo token: (1, C)
  """
  if tuple(x.shape) == tuple(dst_shape):
      return x

  # HF: (1, 1, C) -> repo: (1, C)
  if x.ndim == 3 and x.shape[0] == 1 and x.shape[1] == 1 and tuple(dst_shape) == (1, x.shape[2]):
      return x.squeeze(1)

  # HF: (1, N, C) -> repo: (N, C)
  if x.ndim == 3 and x.shape[0] == 1 and tuple(dst_shape) == tuple(x.shape[1:]):
      return x.squeeze(0)

  # HF: (N, C) -> repo: (1, N, C)
  if x.ndim == 2 and len(dst_shape) == 3 and dst_shape[0] == 1 and tuple(dst_shape[1:]) == tuple(x.shape):
      return x.unsqueeze(0)

  raise RuntimeError(
      f"shape mismatch for {dst_key}: "
      f"HF {src_key} {tuple(x.shape)} vs target {tuple(dst_shape)}"
  )


def put(dst_key, src_key):
  if src_key not in hf:
      raise KeyError(f"missing HF key: {src_key}")
  if dst_key not in target:
      raise KeyError(f"target model does not have key: {dst_key}")

  x = hf[src_key]
  x = adapt_shape(x, target[dst_key].shape, src_key, dst_key)
  converted[dst_key] = x

# embeddings
put("cls_token", "embeddings.cls_token")
put("mask_token", "embeddings.mask_token")
put("storage_tokens", "embeddings.register_tokens")
put("patch_embed.proj.weight", "embeddings.patch_embeddings.weight")
put("patch_embed.proj.bias", "embeddings.patch_embeddings.bias")

# transformer blocks
num_blocks = 12  # vitb16 has 12 blocks

for i in range(num_blocks):
  # norms
  put(f"blocks.{i}.norm1.weight", f"layer.{i}.norm1.weight")
  put(f"blocks.{i}.norm1.bias", f"layer.{i}.norm1.bias")
  put(f"blocks.{i}.norm2.weight", f"layer.{i}.norm2.weight")
  put(f"blocks.{i}.norm2.bias", f"layer.{i}.norm2.bias")

  # attention qkv weight: concat q, k, v along dim=0
  q_w = hf[f"layer.{i}.attention.q_proj.weight"]
  k_w = hf[f"layer.{i}.attention.k_proj.weight"]
  v_w = hf[f"layer.{i}.attention.v_proj.weight"]
  qkv_w = torch.cat([q_w, k_w, v_w], dim=0)

  dst_qkv_w = f"blocks.{i}.attn.qkv.weight"
  if tuple(qkv_w.shape) != tuple(target[dst_qkv_w].shape):
      raise RuntimeError(
          f"shape mismatch for {dst_qkv_w}: "
          f"converted {tuple(qkv_w.shape)} vs target {tuple(target[dst_qkv_w].shape)}"
      )
  converted[dst_qkv_w] = qkv_w

  # attention qkv bias
  # HF DINOv3 usually has q_proj.bias and v_proj.bias, but no k_proj.bias.
  q_b = hf[f"layer.{i}.attention.q_proj.bias"]
  v_b = hf[f"layer.{i}.attention.v_proj.bias"]
  k_b = hf.get(f"layer.{i}.attention.k_proj.bias", torch.zeros_like(q_b))
  qkv_b = torch.cat([q_b, k_b, v_b], dim=0)

  dst_qkv_b = f"blocks.{i}.attn.qkv.bias"
  if tuple(qkv_b.shape) != tuple(target[dst_qkv_b].shape):
      raise RuntimeError(
          f"shape mismatch for {dst_qkv_b}: "
          f"converted {tuple(qkv_b.shape)} vs target {tuple(target[dst_qkv_b].shape)}"
      )
  converted[dst_qkv_b] = qkv_b

  # attention output projection
  put(f"blocks.{i}.attn.proj.weight", f"layer.{i}.attention.o_proj.weight")
  put(f"blocks.{i}.attn.proj.bias", f"layer.{i}.attention.o_proj.bias")

  # layer scale
  put(f"blocks.{i}.ls1.gamma", f"layer.{i}.layer_scale1.lambda1")
  put(f"blocks.{i}.ls2.gamma", f"layer.{i}.layer_scale2.lambda1")

  # MLP
  put(f"blocks.{i}.mlp.fc1.weight", f"layer.{i}.mlp.up_proj.weight")
  put(f"blocks.{i}.mlp.fc1.bias", f"layer.{i}.mlp.up_proj.bias")
  put(f"blocks.{i}.mlp.fc2.weight", f"layer.{i}.mlp.down_proj.weight")
  put(f"blocks.{i}.mlp.fc2.bias", f"layer.{i}.mlp.down_proj.bias")

# Start from target state_dict so buffers like rope_embed.periods and qkv.bias_mask are kept.
out = dict(target)
out.update(converted)

missing_after_convert = [k for k in target.keys() if k not in out]
print("converted keys:", len(converted))
print("target keys:", len(target))
print("missing_after_convert:", missing_after_convert)

# Final sanity check
model.load_state_dict(out, strict=True)
print("strict load into local DINOv3 repo model: OK")

os.makedirs(os.path.dirname(DST), exist_ok=True)
torch.save(out, DST)
print("saved converted pth:", DST)