# openpi-server 环境配置

## 创建 Python 3.11 环境
```
conda create \
  -p /data/Xixixi/VLA1/conda_envs/openpi-server \
  python=3.11 \
  pip \
  -y

conda activate \
  /data/Xixixi/VLA1/conda_envs/openpi-server

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

## 设置 checkpoint 缓存目录
```
mkdir -p \
  /data/Xixixi/VLA1/model_cache/openpi

export OPENPI_DATA_HOME=\
/data/Xixixi/VLA1/model_cache/openpi
```

## 首次启动 π0.5 policy server
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
  /data/Xixixi/VLA1/conda_envs/openpi-server

cd /data/Xixixi/VLA1/openpi

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi05_libero
```

## 首次启动 π0 policy server
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
  --policy.config=pi0_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi0_libero
```