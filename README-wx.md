# 解决响应 timeout
openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py
openpi/src/openpi/serving/websocket_policy_server.py

# 视觉特征 adapter v1
openpi/src/openpi/models_pytorch/feature_adapter.py: **新增**FP32 残差 adapter、规格与临时特征采集
openpi/src/openpi/models_pytorch/pi0_pytorch.py: **修改**添加 adapter 接入点；增加训练图像增强控制参数

## 去除 baseline 与配置采样随机性
openpi/src/openpi/policies/policy.py: **修改**