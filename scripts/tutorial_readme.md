# Evo-RL 采训推说明

本文档前半部分给出可直接执行的脚本流程，后半部分记录排障与实现细节。建议先按第 1~3 节执行，再查阅 debug 章节。

## 1. 数据转换与 `norm` 统计

### 1.1 执行环境

- 在 CPU 机器执行。
- 仓库根目录：`/mnt/nas/wanghao/openpi_05/Evo-RL`

### 1.2 执行命令

```bash
cd /mnt/nas/wanghao/openpi_05/Evo-RL
bash scripts/run_lerobot_dataset_prepare.sh
```

### 1.3 关键参数（`scripts/run_lerobot_dataset_prepare.sh`）

```bash
DATASET_PARENT_NAME="desk_basket_pick"
CONVERT_PARENT_DIR="lerobot/${DATASET_PARENT_NAME}"
CONVERT_REPO_IDS=(
  "basket_pick_0422_lerobot"
  "basket_pick_0422_lerobot_poor"
)
MERGE_OUTPUT_DIR="lerobot_v3/${DATASET_PARENT_NAME}"
MERGE_REPO_ID="basket_pick_0422_s_140_f_9"
```

- `CONVERT_PARENT_DIR`：待转换数据集所在父目录（v2.1）。
- `CONVERT_REPO_IDS`：本次要转换/合并的数据集列表。
- `MERGE_OUTPUT_DIR`：v3.0 合并结果输出目录。
- `MERGE_REPO_ID`：合并后数据集目录名（也用于后续训练 `repo_id`）。


## 2. Value Function 训练与推理

### 2.1 执行环境

- 在 8 卡 GPU 机器执行。

### 2.2 执行命令

```bash
cd /mnt/nas/wanghao/openpi_05/Evo-RL
bash scripts/run_valuefunc_train.sh <RUN_NAME_value_train>

cd /mnt/nas/wanghao/openpi_05/Evo-RL
bash scripts/run_valuefunc_infer.sh <RUN_NAME_value_train>
```

### 2.3 关键参数（`scripts/run_valuefunc_train.sh`）

```bash
DATASET_REPO_ID="desk_basket_pick/basket_pick_0422_s_140_f_9" # "<HF_USERNAME_OR_ORG>/<DATASET_NAME>"
```

- `DATASET_REPO_ID` 需与第 1 节产出的合并数据集一致。
- `<RUN_NAME_value_train>` 建议包含日期/实验标签，便于区分多轮迭代结果。

## 3. Policy 训练

### 3.1 执行环境

- 在 8 卡 GPU 机器执行。

### 3.2 执行命令

```bash
cd /mnt/nas/wanghao/openpi_05/Evo-RL
bash scripts/run_policy_train.sh <RUN_NAME_policy_train>
```

- `<RUN_NAME_policy_train>` 建议与 value 实验命名策略一致，便于追踪同一轮迭代。

## 4. Policy 推理

### 4.1 执行环境

- 单卡GPU，启动websocket_server
- ORIN，启动websocket_client

### 4.2 执行命令

```bash
# ToDo
```

## 5. 数据合并

### 5.1 执行环境

- 在 CPU 机器执行。

### 5.2 执行命令
- 经过value_infer的数据集的value可视化(可选)
```bash
 python scripts/plot_value_on_video.py --dataset /path/to/dataset \\
      --ep 0 --tag recap --smooth-window 9 --chart-alpha 0.22
```

- 删除第一轮生成的complementary_info
```bash
export USR_NAME="wanghao"  # 用户名
export HF_LEROBOT_HOME="/mnt/nas/${USR_NAME}/data/lerobot_v3/"
source /mnt/data/miniconda3/bin/activate evo-rl_wanghao
lerobot-edit-dataset \
  --repo_id desk_basket_operation/basket_pick_plc_a02 \
  --operation.type=remove_feature \
  --operation.feature_names="['complementary_info.value_recap','complementary_info.advantage_recap','complementary_info.acp_indicator_recap']"
```

- 将新收集的第二轮数据与第一轮合并
```bash
export USR_NAME="wanghao"  # 用户名
export HF_LEROBOT_HOME="/mnt/nas/${USR_NAME}/data/lerobot_v3/"
source /mnt/data/miniconda3/bin/activate evo-rl_wanghao
lerobot-edit-dataset \
  --repo_id=desk_basket_operation/basket_pick_plc_a02 \
  --operation.type=merge \
  --operation.repo_ids="['desk_basket_operation/basket_pick_plc_a02_0515','desk_basket_place/basket_place_a02_03_s_76_f_28']"
```

## 6. Iterative Training Loop（抽象流程）

```text
[Multi-task demonstration data pool]
        |
        v
[Offline RL pretraining for a vision-language-action policy]
        |
        v
[Task-specific initialization / fine-tuning from demonstrations]
        |
        v
|---- Iteration k = 1..K -------------------------------------|
| 1) Deploy current policy π_k and collect new rollout data   |
| 2) Merge into data pool: D <- D U new_data                  |
| 3) Train value function on D                                |
| 4) Infer advantage and binarize into indicator tags         |
| 5) Train advantage-conditioned policy to get π_{k+1}        |
|-------------------------------------------------------------|
        |
        v
[Stronger policy with improved success rate and throughput]
```