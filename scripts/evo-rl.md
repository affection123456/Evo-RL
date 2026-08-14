# Evo-RL 使用与部署说明

本文是当前仓库的数据准备、EE state/action 合同、value/policy 训练及 OpenPI
真机推理的统一说明。旧的 0721/0804 部署和 debug 文档已合并到这里；命令以当前
CLI 和当前代码为准。

仓库路径：

```text
/mnt/nas/wanghao/project/Evo-RL
```

默认环境：

```text
Python: /mnt/data/miniconda3/envs/evo-rl_wanghao/bin/python
MODEL_ZOO: /mnt/data/modelzoo
HF_LEROBOT_HOME: /mnt/nas/datasets/rldata/lerobot
```

## 1. Evo-RL 流程

Evo-RL 当前使用离线 ACP（Advantage-Conditioned Policy）流程：

```text
采集 rollout
  → 训练 value function
  → 推理 value / advantage / indicator
  → 使用 indicator 训练 policy
```

policy 仍使用离线数据上的 flow-matching/模仿学习目标，不是 PPO、SAC 这类边与
真机交互边更新参数的在线强化学习。

训练默认使用：

```text
acp.enable=true
acp.indicator_field=complementary_info.acp_indicator_recap
```

纯 BC 对照可传：

```text
--acp-enable=false
```

## 2. 数据集版本与目录

Python 包版本（例如 `lerobot==0.1.0`）和数据集
`meta/info.json` 中的 `codebase_version` 是两个概念。是否迁移只看
`codebase_version`：

- 旧数据通常是 `v2.1`。
- 当前训练数据应为 `v3.0`。
- v3 数据通常使用 `data/chunk-*/file-*.parquet` 和
  `videos/<camera>/chunk-*/file-*.mp4`。
- parquet 数量和视频文件数量不必一一对应；episode/视频对应关系由
  `meta/episodes` 中的时间戳和文件索引确定。

本地 v2.1→v3.0 的封装实现位于：

```text
src/lerobot/datasets/convert_local_lerobot_v21_to_v30.py
```

通常不直接调用它，统一使用 prepare shell。

## 3. 原始 EE state/action 布局

原始 A2D EE 向量前 26 维：

```text
[0:3]    左臂 xyz
[3:7]    左臂 quaternion (x,y,z,w)
[7:10]   右臂 xyz
[10:14]  右臂 quaternion (x,y,z,w)
[14:20]  左夹爪/灵巧手 6 个通道
[20:26]  右夹爪/灵巧手 6 个通道
```

常见 44D `ee_state`：

```text
[26:28]  head joints
[28:32]  waist joints
[32:44]  end wrench
```

常见 34D `ee_actions`：

```text
[26:28]  head command
[28:32]  waist command
[32:34]  velocity extras
```

磁盘保存的是绝对 quaternion EE 数据。Rot6D、维度选择、pad 和归一化均在
stats/preprocessor 中完成。

夹爪数据应处于 `[0,1]`。prepare 默认会检查 `[14:26]`，原始
`[0,1000]` 数据会除以 1000；已经处于 `[0,1]` 的数据不会重复缩放。

## 4. 模型侧 full32 EE 布局

PI05、PI0-DMP 和 Pistar06 使用同一历史双臂 Rot6D 顺序：

```text
left_xyz(3), left_Rot6D(6), right_xyz(3), right_Rot6D(6), raw_tail
```

转换后统一 clip/pad 到 32D；pose delta 仅作用于前 18 维，raw tail 保持绝对值。
PI05 和 PI0-DMP 的 flow loss 覆盖全部 32 维。

Rot6D 使用旋转矩阵前两列：

```text
rot6d = [R[:,0], R[:,1]]
```

推理 decode 通过 Gram-Schmidt 恢复旋转矩阵，再转回 xyzw quaternion。

### 4.1 Stats 必须与训练布局一致

prepare stats 使用相同的双臂 full32 转换和前 18 维 pose delta。改变
delta/absolute 模式后必须重新 augment stats，并使用新 run 名训练。

## 5. 数据准备

进入仓库：

```bash
cd /mnt/nas/wanghao/project/Evo-RL
```

### 5.1 合并 magazine slot pick/place

```bash
bash scripts/run_lerobot_dataset_prepare.sh \
  --dataset-repo-id=magazine_slot_operation_200807 \
  --sources="magazine_slot_pick_20260807_0 magazine_slot_place_20260807_0"
```

默认执行：

```text
convert/merge → gripper normalize → dual-arm full32 Rot6D stats → dataset report
```

### 5.2 仅重新计算 magazine shelf stats

```bash
bash scripts/run_lerobot_dataset_prepare.sh augment \
  --dataset-repo-id=magazine_shelf_pick_20260807_0
```

关闭夹爪尺度处理：

```text
--normalize-ee-gripper=false
```

一般不应关闭；该操作本身是幂等的。

### 5.3 PI0-DMP 数据

PI0-DMP 数据位于其他根目录时显式指定：

```bash
bash scripts/run_lerobot_dataset_prepare_pi0_dmp.sh augment \
  --hf-lerobot-home=/mnt/nas/datasets/rldata/lerobot_with_ref \
  --dataset-repo-id=dmp_data_recap/unt_merged_Mz_right_pik_DMP_lerobot
```

PI0-DMP 会额外为以下字段生成同一合同空间的 stats：

```text
observation.state
observation.reference.state
action
observation.ref_actions
```

## 6. 三种 PRESET

### 6.1 `pi05_data_lerobotv3`

- policy：PI05
- 数据：普通 LeRobot v3
- 当前 EE state + 当前相机
- 不需要 reference state/action/image

### 6.2 `pi05_data_dmp`

- policy：PI05
- 数据：DMP 数据，但 PI05 不消费 reference
- 支持 DMP bare keys rename
- 不需要把 reference 输入送给 PI05 bridge

### 6.3 `pi0_dmp_data_dmp`

- policy：PI0-DMP
- 当前和参考 EE state
- 当前和参考图像
- reference action chunk

三个 PRESET 默认均使用：

```text
absolute action
QUANTILES
pad32
left xyz+Rot6D, right xyz+Rot6D, raw tail
pose delta（如启用）仅作用于前 18 维
```

## 7. Value 训练与推理

value 训练默认 batch size 为 16。位置参数数字不再表示 batch；只有
`--batch-size=` 生效。

PI05 + LeRobot v3：

```bash
bash scripts/run_valuefunc_train.sh 0811_pi05_v3 \
  --dataset-repo-id=magazine_shelf_pick_20260807_0

bash scripts/run_valuefunc_infer.sh 0811_pi05_v3 \
  --dataset-repo-id=magazine_shelf_pick_20260807_0
```

PI05 + DMP：

```bash
bash scripts/run_valuefunc_train.sh 0731_pi05_dmp \
  --preset=pi05_data_dmp \
  --hf-lerobot-home=/mnt/nas/datasets/rldata/lerobot_with_ref \
  --dataset-repo-id=dmp_data_recap/unt_merged_Mz_right_pik_DMP_lerobot

bash scripts/run_valuefunc_infer.sh 0731_pi05_dmp \
  --preset=pi05_data_dmp \
  --hf-lerobot-home=/mnt/nas/datasets/rldata/lerobot_with_ref \
  --dataset-repo-id=dmp_data_recap/unt_merged_Mz_right_pik_DMP_lerobot
```

PI0-DMP + DMP：

```bash
bash scripts/run_valuefunc_train.sh 0731_pi0_dmp \
  --preset=pi0_dmp_data_dmp \
  --hf-lerobot-home=/mnt/nas/datasets/rldata/lerobot_with_ref \
  --dataset-repo-id=dmp_data_recap/unt_merged_Mz_right_pik_DMP_lerobot

bash scripts/run_valuefunc_infer.sh 0731_pi0_dmp \
  --preset=pi0_dmp_data_dmp \
  --hf-lerobot-home=/mnt/nas/datasets/rldata/lerobot_with_ref \
  --dataset-repo-id=dmp_data_recap/unt_merged_Mz_right_pik_DMP_lerobot
```

## 8. Policy 训练

PI05 + magazine shelf，默认双臂 full32 Rot6D：

```bash
bash scripts/run_policy_train.sh 0813_pi05_magazine_shelf_right \
  --dataset-repo-id=magazine_shelf_pick_20260807_0 \
  --acp-enable=false
```

PI05 + DMP：

```bash
bash scripts/run_policy_train.sh 0731_pi05_dmp \
  --preset=pi05_data_dmp \
  --hf-lerobot-home=/mnt/nas/datasets/rldata/lerobot_with_ref \
  --dataset-repo-id=dmp_data_recap/unt_merged_Mz_right_pik_DMP_lerobot
```

PI0-DMP + DMP：

```bash
bash scripts/run_policy_train.sh 0731_pi0_dmp \
  --preset=pi0_dmp_data_dmp \
  --hf-lerobot-home=/mnt/nas/datasets/rldata/lerobot_with_ref \
  --dataset-repo-id=dmp_data_recap/unt_merged_Mz_right_pik_DMP_lerobot
```

训练脚本发现同名 `outputs/train/<RUN_NAME>` 时会删除后重新训练，因此正式实验必须
使用新的 run 名。

## 9. 多 GPU

训练脚本会自动检测 GPU 数量。也可显式指定：

```text
--num-gpus=8
--use-multi-gpu=1
--gpu-id-list=0,1,2,3,4,5,6,7
```

多卡由 `accelerate launch` 启动多个进程，每个进程解析同一套训练参数。
`batch_size` 是每个进程/每张卡的 batch，因此：

```text
global batch ≈ per-GPU batch × num processes
```

学习率不会自动按总 batch 缩放。

## 10. OpenPI 真机推理

### 10.1 启动

PI05 + LeRobot v3：

```bash
bash scripts/run_policy_infer_openpi_bridge.sh RUN_NAME \
  --port=8002
```

PI05 + DMP：

```bash
bash scripts/run_policy_infer_openpi_bridge.sh RUN_NAME \
  --port=8002 \
  --preset=pi05_data_dmp
```

PI0-DMP：

```bash
bash scripts/run_policy_infer_openpi_bridge.sh RUN_NAME \
  --port=8002 \
  --preset=pi0_dmp_data_dmp
```

推理从 checkpoint 加载 Rot6D 配置。不要在部署时覆盖训练时的表示设置。

### 10.2 PI05 输入

OpenPI 客户端输入映射：

```text
observation/state              → observation.ee_state
observation/image              → observation.images.top_head
observation/right_wrist_image  → observation.images.hand_right
observation/left_wrist_image   → observation.images.hand_left（可选）
```

客户端发送至少 14D 的完整双臂 raw quaternion EE state。模型输入按历史顺序转换为
`left xyz+Rot6D, right xyz+Rot6D, raw tail` 后 clip/pad32；输出再恢复为 raw quaternion action。

### 10.3 PI0-DMP 输入

```text
observation/state
reference/state
ref_actions
observation/image
observation/right_wrist_image
reference/image
reference/right_wrist_image
```

rename 后 policy 需要：

```text
observation.state
observation.reference.state
observation.ref_actions
observation.images.top_head
observation.images.hand_right
observation.images.ref_top_head
observation.images.ref_hand_right
```

PI0-DMP 当前/参考 state、当前/参考 action 均使用同一个双臂 full32 Rot6D 布局。

### 10.4 服务自检日志

首次请求应看到：

```text
First OpenPI observation after rename ...
Model-facing EE layout: dual-arm xyz+Rot6D pose plus raw tail, clip/pad32 ...
```

常见错误：

- `Missing observation.ee_state`：PI05 rename_map 错误，state 被映射到了
  `observation.state`。
- `D>=14`：客户端没有发送完整双臂 xyz+quaternion raw EE state。
- action/姿态离谱：优先核对 checkpoint 与 stats 的 full32 布局、夹爪尺度和 quat 顺序。
- reference 字段缺失：当前 checkpoint 实际是 PI0-DMP。

## 11. 推理性能与安全默认

PI0-DMP 有四路图像、reference state/action，并进行多步 flow 去噪，计算量明显高于
普通双相机 PI05。历史真机测量中单次模型推理约 2.3–2.6 秒，端到端约
3.3–5.2 秒；实际延迟还受 GPU 争用、网络、图像编码和客户端调度影响。

安全默认：

```text
num_inference_steps=10
PI0_DMP_KV_CACHE=false
chunk_size=50
```

不要未经验证直接把 chunk size 从 50 改为 30；checkpoint、attention mask 和
reference action horizon 都与 chunk size 绑定。

PI0-DMP KV-cache 是实验功能：

```text
--pi0_dmp_kv_cache=true
```

它会复用图像/语言 prefix，但历史测试曾出现停止姿态扭曲，尚未证明和原始 full
forward 数值等价。需要测试时必须保持同一输入并分开比较：

```text
A: 10 steps，cache off
B: 10 steps，cache on
C: 5 steps，cache off
```

真机出现姿态扭曲、动作突跳或步长超限时，立即恢复 10 steps、关闭 cache，并先做
低速/空载或离线 action 检查。

## 12. 旧 checkpoint

`outputs/train/0721_1` 等旧 PI0-DMP checkpoint 使用 34D delta processor，
不等同于当前 full32 absolute 配置。

部署旧 checkpoint 时：

- 使用 checkpoint 自带 config/preprocessor/stats。
- 不要用当前默认参数覆盖其处理器配置。
- 不要把旧 34D stats 与当前 full32 训练混用。
- 若 Rot6D 编码来自标准列布局修复之前，需要重算 stats 并重训，不能只改 decode。

新实验建议使用新 run 名和当前统一合同重新 augment、训练。

## 13. 离线模型与 tokenizer

训练脚本默认：

```text
MODEL_ZOO=/mnt/data/modelzoo
```

常用本地资源：

```text
/mnt/data/modelzoo/google/paligemma-3b-pt-224
/mnt/data/modelzoo/google/gemma-3-270m
/mnt/data/modelzoo/google/siglip-so400m-patch14-384
```

如果训练正常、直接推理却尝试联网下载 tokenizer，通常是推理 shell 没有设置
`MODEL_ZOO`。统一使用仓库 shell 脚本可避免该问题。

Gemma 属于 gated model；首次下载仍需要账号同意协议和有效 token。稳定部署应提前
下载到本地，不依赖运行时网络或镜像。

## 14. 关键文件

```text
scripts/lib_exp_preset.sh
scripts/run_lerobot_dataset_prepare.sh
scripts/run_lerobot_dataset_prepare_pi0_dmp.sh
scripts/run_valuefunc_train.sh
scripts/run_valuefunc_infer.sh
scripts/run_policy_train.sh
scripts/run_policy_infer_openpi_bridge.sh

src/lerobot/policies/pi05/processor_pi05.py
src/lerobot/policies/pi0_dmp/processor_pi0_dmp.py
src/lerobot/values/pistar06/processor_pistar06.py
src/lerobot/datasets/v30/augment_dataset_quantile_stats.py
src/lerobot/scripts/lerobot_policy_infer.py
src/lerobot/scripts/lerobot_policy_infer_openpi_bridge.py
```

## 15. 最小检查清单

数据准备前：

- `meta/info.json` 是 v3.0，或使用 prepare 转换。
- 夹爪 `[14:26]` 是 `[0,1]`，或保留默认归一化。
- prepare 的 EE 合同与计划训练完全一致。

训练前：

- `meta/stats.json` 已按当前 full32 布局重算。
- value 和 policy 使用相同的双臂 Rot6D 布局。
- 新实验使用新的 run 名。

部署前：

- checkpoint 类型和 PRESET 一致。
- 客户端发送完整双臂 raw quaternion EE state。
- PI05 不要求 reference；PI0-DMP 必须提供 reference。
- 默认 10 个去噪步骤且关闭实验 KV-cache。
