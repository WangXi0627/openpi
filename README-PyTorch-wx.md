# openpi-server 环境配置

## 创建 Python 3.11 环境
TORCH_SERVER_ENV="/data/Xixixi/VLA1/conda_envs/openpi-server-torch"
```
conda create \
  -p "/data/Xixixi/VLA1/conda_envs/openpi-server-torch" \
  python=3.11 \
  pip \
  -y

conda activate \
  /data/Xixixi/VLA1/conda_envs/openpi-server-torch

python -m pip install \
  --upgrade \
  pip \
  uv
```

## 安装官方 OpenPI
```
cd /data/Xixixi/VLA1/openpi

export UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX}"\

GIT_LFS_SKIP_SMUDGE=1 \
uv sync --active

GIT_LFS_SKIP_SMUDGE=1 \
uv pip install \
  --python "${CONDA_PREFIX}/bin/python" \
  -e .
```

## 应用官方 Transformers patch
```
# 1.取得当前环境的 Transformers 路径
export TRANSFORMERS_DIR="$(
python - <<'PY'
from pathlib import Path
import transformers

print(Path(transformers.__file__).resolve().parent)
PY
)"

echo "${TRANSFORMERS_DIR}"


# 2.应用 patch
cp -r \
  "/data/Xixixi/VLA1/openpi/src/openpi/models_pytorch/transformers_replace/." \
  "${TRANSFORMERS_DIR}/"


# 3.测试导入
python - <<'PY'
import torch
import transformers
from openpi.models_pytorch import pi0_pytorch

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("pi0_pytorch:", pi0_pytorch.__file__)
print("OpenPI PyTorch import passed.")
PY
```

## 转换 pi05_libero JAX checkpoint
```
# 1.找到 JAX checkpoint
export JAX_CHECKPOINT=/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi05_libero


# 2.确定输出目录
export TORCH_CHECKPOINT=/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch


# 3.正式转换
CUDA_VISIBLE_DEVICES="" \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir "${JAX_CHECKPOINT}" \
  --config_name pi05_libero \
  --output_path "${TORCH_CHECKPOINT}" \
  --precision bfloat16


# 4.确保 assets 被复制
mkdir -p "${TORCH_CHECKPOINT}/assets"

cp -a \
  "${JAX_CHECKPOINT}/assets/." \
  "${TORCH_CHECKPOINT}/assets/"
```

## 启动 π0.5 policy server (PyTorch)
```
conda activate \
  /data/Xixixi/VLA1/conda_envs/openpi-server

cd /data/Xixixi/VLA1/openpi

export OPENPI_DATA_HOME=\
/data/Xixixi/VLA1/model_cache/openpi
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

## 启动 π0.5 policy server
```
conda activate \
  /data/Xixixi/VLA1/conda_envs/openpi-server-torch

cd /data/Xixixi/VLA1/openpi

export TORCH_CHECKPOINT=/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch

CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir="${TORCH_CHECKPOINT}"
```

## 转换 pi0_libero JAX checkpoint
```
# 1.找到 JAX checkpoint
export JAX_CHECKPOINT=/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi0_libero


# 2.确定输出目录
export TORCH_CHECKPOINT=/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch


# 3.正式转换
CUDA_VISIBLE_DEVICES="" \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir "${JAX_CHECKPOINT}" \
  --config_name pi0_libero \
  --output_path "${TORCH_CHECKPOINT}" \
  --precision bfloat16


# 4.确保 assets 被复制
mkdir -p "${TORCH_CHECKPOINT}/assets"

cp -a \
  "${JAX_CHECKPOINT}/assets/." \
  "${TORCH_CHECKPOINT}/assets/"
```