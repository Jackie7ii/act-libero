# ACT-LIBERO

ACT (Action Chunking with Transformers) 在 LIBERO 仿真环境上的独立实现。

从 [openvla-oft](https://github.com/moojink/openvla-oft) 中提取，去除所有对原项目的依赖，可作为独立项目运行。

---

## 目录

- [文件结构](#文件结构)
- [依赖安装](#依赖安装)
- [数据准备](#数据准备)
- [配置参数详解](#配置参数详解)
- [训练](#训练)
- [评估](#评估)
- [模型架构](#模型架构)
- [状态与动作空间](#状态与动作空间)
- [Temporal Aggregation](#temporal-aggregation)
- [WandB 实验追踪](#wandb-实验追踪)
- [Checkpoint 说明](#checkpoint-说明)

---

## 文件结构

```
act-libero/
├── runact_libero.py          # 训练 + 评估主入口
├── config.py                 # 所有超参数与路径配置
├── datasetdeal.py            # RLDS 数据集加载、缓存、DataLoader
├── policy.py                 # ACTPolicy / CNNMLPPolicy 定义
├── libero_utils.py           # LIBERO 环境工具函数（图像提取、视频保存等）
├── __init__.py               # 包入口（供外部引用）
├── detr/
│   ├── detr_vae.py           # DETRVAE CVAE 主体模型
│   ├── backbone.py           # ResNet 视觉骨干网络（基于 torchvision）
│   ├── transformer.py        # Transformer 编码器 / 解码器
│   └── position_encoding.py  # 正弦 / 学习式位置编码
├── utils/
│   └── check_npz.py          # 查看 npz 缓存文件内容的调试工具
├── ckpt/                     # checkpoint 默认保存目录
└── requirements.txt          # Python 依赖列表
```

---

## 依赖安装

### 1. Python 环境

推荐 Python 3.8 或以上版本。

### 2. 安装 Python 包

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含：

```
torch
torchvision
numpy
tqdm
wandb
imageio
imageio-ffmpeg
tensorflow
tensorflow-datasets
libero
```

> **注意**：`torch` 需要与 CUDA 版本匹配。建议先手动安装对应版本的 PyTorch，再执行上述命令，避免自动安装 CPU 版本。例如 CUDA 11.8：
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 3. 安装 LIBERO

LIBERO 提供仿真环境和 benchmark。安装方式：

```bash
# 方式一：pip 安装（若已包含在 requirements.txt 中）
pip install libero

# 方式二：从源码安装（推荐，获取最新版本）
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

LIBERO 依赖 **MuJoCo**，首次使用时会自动下载。若遇到 MuJoCo 许可证问题，请参考 [MuJoCo 官方文档](https://mujoco.org/)。

### 4. 配置 WandB

训练和评估均使用 WandB 记录指标。首次使用需登录：

```bash
wandb login
```

如希望离线运行（不上传数据）：

```bash
wandb offline
```

---

## 数据准备

### 下载数据集

本项目使用托管在 HuggingFace 上的 [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds) 数据集，包含四个任务套件，共约 **10 GB**：

| 数据集名称 | 任务套件 | 任务数 |
|---|---|---|
| `libero_spatial_no_noops` | LIBERO-Spatial（空间关系任务） | 10 |
| `libero_object_no_noops` | LIBERO-Object（物体操作任务） | 10 |
| `libero_goal_no_noops` | LIBERO-Goal（目标导向任务） | 10 |
| `libero_10_no_noops` | LIBERO-10 / LIBERO-Long（长视域任务） | 10 |

> `_no_noops` 表示已过滤掉训练数据中近零动作（no-op）的帧。

**下载方式（需要 git-lfs）**：

```bash
# 安装 git-lfs（如果尚未安装）
git lfs install

# 克隆整个数据集（包含全部四个任务套件，~10 GB）
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

下载完成后，目录结构如下：

```
modified_libero_rlds/
├── libero_spatial_no_noops/
├── libero_object_no_noops/
├── libero_goal_no_noops/
└── libero_10_no_noops/
```

将 `config.py` 中的 `data_root` 设置为该目录的路径，`task_suite_name` 设置为对应子目录名即可。

---

### 数据格式

本项目使用 **RLDS（TFRecord）格式** 的 LIBERO 数据集，通过 `tensorflow_datasets` 加载。

数据集来源为 [openvla-oft](https://github.com/moojink/openvla-oft) 项目中经过预处理的 LIBERO RLDS 数据（`modified_libero_rlds`）。

数据集中每条 episode 需包含以下字段：

| 字段 | 说明 | 形状 |
|---|---|---|
| `observation/image` | 第三人称（agent view）相机图像 | `(256, 256, 3)` uint8 |
| `observation/wrist_image` | 腕部相机图像 | `(256, 256, 3)` uint8 |
| `observation/state` | 机器人本体状态（qpos，8 维） | `(8,)` float32 |
| `action` | 机器人动作（7 维） | `(7,)` float32 |

### 数据路径配置

在 `config.py` 中修改：

```python
'data_root':       '/path/to/your/modified_libero_rlds',  # RLDS 数据集根目录
'task_suite_name': 'libero_spatial_no_noops',             # 数据集名称（对应 tfds 数据集目录名）
```

> `task_suite_name` 中的 `_no_noops` 后缀仅用于数据加载，Benchmark 查找时会自动去除（即实际使用 `libero_spatial`）。

### 自动缓存机制

**首次运行**时，程序从 TFRecord 逐步解析数据，并将每条 episode 保存为 `.npz` 文件：

```
{data_root}/{task_suite_name}_cache/
├── episode_0000.npz
├── episode_0001.npz
└── ...
```

**后续运行**直接从 `.npz` 缓存加载，跳过 TFRecord 解析，速度显著更快。

如需重新生成缓存（例如数据更新后），删除缓存目录即可：

```bash
rm -rf /path/to/data_root/libero_spatial_no_noops_cache/
```

每个 `.npz` 文件包含四个数组：`images`、`wrist_images`、`qpos`、`actions`。

### 数据集划分

数据在运行时随机划分（8:2）：
- **80%** 用于训练
- **20%** 用于验证

划分基于 episode 索引，每次运行的划分结果由 `seed` 参数控制。

---

## 配置参数详解

所有参数统一在 `config.py` 的 `ACT_LIBERO_CONFIG` 字典中配置。

### 数据参数

| 参数 | 说明 | 当前默认值 |
|---|---|---|
| `data_root` | RLDS 数据集根目录 | `/local_data/.../modified_libero_rlds` |
| `task_suite_name` | tfds 数据集名称 | `libero_spatial_no_noops` |
| `camera_names` | 使用的相机列表（顺序即模型输入顺序） | `['images', 'wrist_images']` |

相机名称映射关系（定义于 `libero_utils.py`）：

| `camera_names` 中的名称 | 对应的环境观测键 |
|---|---|
| `images` | `obs['agentview_image']`（第三人称视角） |
| `wrist_images` | `obs['robot0_eye_in_hand_image']`（腕部相机） |

### 模型参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `backbone` | 视觉骨干网络 | `resnet18` |
| `position_embedding` | 位置编码类型 | `sine` |
| `hidden_dim` | Transformer 隐层维度 | `256` |
| `dim_feedforward` | FFN 维度 | `2048` |
| `nheads` | 注意力头数 | `8` |
| `enc_layers` | CVAE 编码器层数 | `4` |
| `dec_layers` | Transformer 解码器层数 | `6` |
| `pre_norm` | 是否使用 Pre-LayerNorm | `False` |
| `dropout` | Dropout 比例 | `0.1` |
| `masks` | 是否使用分割 mask | `False` |
| `dilation` | 是否使用空洞卷积 | `False` |
| `num_queries` | Action chunk 大小（预测步数） | `75` |
| `action_dim` | 动作空间维度 | `7` |
| `qpos_dim` | 机器人状态维度 | `8` |

### 训练参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `lr` | 主干网络以外参数的学习率 | `1e-5` |
| `lr_backbone` | Backbone 学习率 | `1e-6` |
| `weight_decay` | AdamW 权重衰减 | `1e-4` |
| `kl_weight` | KL 散度损失权重 | `10` |
| `batch_size` | 训练与验证的 batch 大小 | `8` |
| `num_epochs` | 总训练轮数 | `2500` |
| `seed` | 随机种子（影响数据划分） | `42` |
| `gpu_id` | 使用的 GPU 编号 | `5` |

### 评估参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `temporal_agg` | 是否启用 Temporal Aggregation | `False` |
| `agg_k` | Temporal Aggregation 指数衰减系数 | `0.5` |
| `num_open_loop_steps` | Open-loop 模式下每次 query 执行的步数 | `8` |
| `num_steps_wait` | 每个 episode 开始时等待的空动作步数 | `10` |
| `eval_seed` | Rollout 评估随机种子 | `42` |
| `num_trials_per_task_final` | 最终评估时每个 task 的试验次数 | `20` |

### 训练中评估参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `rollout_eval_freq` | 每隔多少 epoch 做一次 rollout 评估（0 = 禁用） | `100` |
| `num_trials_per_task` | 训练中 rollout 评估每个 task 的试验次数 | `5` |

### 保存参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `ckpt_dir` | Checkpoint 保存目录 | `./ckpt/act_libero_spatial0` |
| `video_dir` | Rollout 视频保存目录 | `./ckpt/act_libero_spatial0` |
| `save_every` | 每隔多少 epoch 保存一次周期性 checkpoint | `50` |

### 各任务套件最大步数（`TASK_MAX_STEPS`）

| 任务套件 | 最大步数 |
|---|---|
| `libero_spatial` | 220 |
| `libero_object` | 280 |
| `libero_goal` | 300 |
| `libero_10` | 520 |
| `libero_90` | 400 |

---

## 训练

### 确认配置

1. 确认 `config.py` 中的 `data_root` 指向正确的数据目录
2. 确认 `gpu_id` 与服务器可用 GPU 匹配
3. 确认 `ckpt_dir` 目录有写权限（程序会自动创建）

### 启动训练

```bash
cd /local_data/jl17265/projects/act-libero
python runact_libero.py
```

### 训练输出格式

```
Using device: cuda:5
[data] loading from cache: /path/to/cache
Epoch 0: train=x.xxxx, val=x.xxxx, best_val=x.xxxx
Epoch 1: train=x.xxxx, val=x.xxxx, best_val=x.xxxx
...
```

当 `rollout_eval_freq > 0` 时，每隔指定 epoch 还会输出 rollout 评估结果：

```
[Epoch 100] Running rollout eval (5 trials/task)...
  Task 0 (pick up the alphabet soup ...): 4/5 = 0.8000
  Task 1 (move the milk ...): 3/5 = 0.6000
  ...
[Epoch 100] Rollout SR: 0.7000 (best so far: -1.0000)
[Epoch 100] New best rollout! SR=0.7000 -> policy_best_rollout.ckpt saved
```

### 训练结束输出

```
========== Training finished ==========
[Val-loss ckpt]  epoch 1842, val_loss 0.012345
[Rollout ckpt]   best rollout SR = 0.7800
Eval with val-loss ckpt : --ckpt_name policy_best.ckpt
Eval with rollout ckpt  : --ckpt_name policy_best_rollout.ckpt
=======================================
```

### 损失函数

ACT 使用 CVAE 框架，总损失为：

```
loss = L1_reconstruction_loss + kl_weight × KL_divergence
```

- **L1 loss**：预测动作与真实动作之间的 L1 距离（对 padding 位置不计算）
- **KL loss**：CVAE 隐变量分布与标准正态分布的 KL 散度
- 两者均同步记录到 WandB

---

## 评估

### 使用 val-loss 最优 checkpoint 评估（默认）

```bash
python runact_libero.py --eval
```

### 指定特定 checkpoint

```bash
python runact_libero.py --eval --ckpt_name policy_best_rollout.ckpt
python runact_libero.py --eval --ckpt_name checkpoint_epoch_500.ckpt
python runact_libero.py --eval --ckpt_name policy_last.ckpt
```

### 评估流程

对 task suite 中的每个 task：
1. 加载对应的 LIBERO 仿真环境
2. 运行 `num_trials_per_task_final`（默认 20）次 rollout
3. 每次 rollout 先执行 `num_steps_wait`（默认 10）步空动作，待环境稳定
4. 根据 `temporal_agg` 配置选择动作执行方式（见下文）
5. 保存 rollout 视频到 `video_dir/video/` 目录

### 评估输出格式

```
Task 0: pick up the alphabet soup and place it in the basket
Trial 1: SUCCESS | 1/1
Trial 2: FAIL    | 1/2
...
Task 0 success rate: 0.7500

Overall success rate: 0.7200 (144/200)
```

### 视频命名格式

```
{video_dir}/video/{datetime}--act--episode={N}--success={True/False}--task={task_name}.mp4
```

---

## 模型架构

ACT 基于 CVAE + DETR 风格的 Transformer，包含以下组件：

```
输入：qpos (8维) + 多路相机图像 (N × 3 × 256 × 256)
         │
    ┌────┴────┐
    │ 训练时  │
    │ CVAE    │  ← encoder: [CLS, qpos, action_seq] → latent z (32维)
    │ encoder │
    └────┬────┘
         │ latent z (或推理时的零向量)
         ↓
┌─────────────────────┐
│  ResNet-18 Backbone │  ← 提取每路相机特征图
└─────────┬───────────┘
          │ 多路特征图拼接 (width 方向)
          ↓
┌─────────────────────┐
│  Transformer 解码器 │  ← 多头注意力 (6层, 8头)
│  (基于 DETR)        │    查询数 = num_queries = 75
└─────────┬───────────┘
          │
          ↓
输出：预测动作序列 (75 × 7)
```

**CVAE latent 维度**：32
**模型总参数**：约 85M（取决于相机数量）

---

## 状态与动作空间

### 机器人状态（qpos，8 维）

由以下三部分拼接（定义于 `runact_libero.py`）：

| 分量 | 来源 | 维度 |
|---|---|---|
| 末端执行器位置 | `obs['robot0_eef_pos']` | 3 |
| 末端执行器朝向（轴角表示） | `quat2axisangle(obs['robot0_eef_quat'])` | 3 |
| 夹爪 qpos | `obs['robot0_gripper_qpos']` | 2 |

> 四元数 `(x, y, z, w)` 在模型输入前转换为轴角表示 `(ax, ay, az)`。

### 动作（7 维）

| 分量 | 说明 | 维度 |
|---|---|---|
| 末端执行器位移 | delta eef position | 3 |
| 末端执行器朝向变化 | delta axis-angle | 3 |
| 夹爪控制 | gripper open/close | 1 |

### 归一化

训练时对状态和动作进行 z-score 归一化，统计量（mean/std）从**训练集**计算，保存在 checkpoint 的 `norm_stats` 字段中，评估时自动读取。

```python
norm_stats = {
    'action_mean': ...,  # shape (7,)
    'action_std':  ...,  # shape (7,)
    'state_mean':  ...,  # shape (8,)
    'state_std':   ...,  # shape (8,)
}
```

---

## Temporal Aggregation

在 `config.py` 中通过 `temporal_agg` 切换：

### Open-loop 模式（`temporal_agg: False`，默认）

```python
'temporal_agg':        False,
'num_open_loop_steps': 8,   # 每次 query 执行 8 步动作
```

每次 query 生成 75 步动作序列，取前 `num_open_loop_steps` 步依次执行，队列耗尽后再次 query。适合计算资源有限的场景，推理频率低。

### Temporal Aggregation 模式（`temporal_agg: True`）

```python
'temporal_agg': True,
'agg_k':        0.5,   # 指数衰减系数，越大历史权重衰减越快
```

每一步都 query 一次模型，对当前时刻的所有历史预测进行指数加权平均：

```
weight[i] = exp(-k × (current_step - i))
action = weighted_average(all_predictions_for_current_step)
```

动作更平滑，但推理开销更大（每步一次前向传播）。

---

## WandB 实验追踪

训练和评估均自动记录到 WandB 项目 `act-libero`。

### 训练记录的指标

| 指标 | 说明 |
|---|---|
| `train_loss` | 训练集总损失（L1 + kl_weight × KL） |
| `val_loss` | 验证集总损失 |
| `train_l1` | 训练集 L1 重建损失 |
| `train_kl` | 训练集 KL 散度 |
| `lr` | 当前主干学习率 |
| `lr_backbone` | 当前 backbone 学习率 |
| `rollout/success_rate` | Rollout 成功率（每隔 `rollout_eval_freq` 记录一次） |

### 评估记录的指标

| 指标 | 说明 |
|---|---|
| `eval/task_{i}_success_rate` | 第 i 个 task 的成功率 |
| `eval/final_success_rate` | 所有 task 的整体成功率 |

---

## Checkpoint 说明

训练过程中会保存以下 checkpoint（均位于 `ckpt_dir`）：

| 文件名 | 保存时机 | 说明 |
|---|---|---|
| `checkpoint_epoch_{N}.ckpt` | 每隔 `save_every` 个 epoch | 包含完整训练状态（model + optimizer + norm_stats） |
| `policy_best.ckpt` | val_loss 创新低时 | 仅含 model 权重 + norm_stats，**推荐用于评估** |
| `policy_best_rollout.ckpt` | rollout 成功率创新高时 | 仅含 model 权重 + norm_stats + rollout_success_rate |
| `policy_last.ckpt` | 训练结束时 | 最后一个 epoch 的 model 权重 + norm_stats |

### 恢复训练

当前不支持从周期性 checkpoint 直接恢复训练，如需此功能需手动修改 `train_bc` 函数以加载 `optimizer_state_dict`。

---

## 常见问题

**Q: 运行时报 `ModuleNotFoundError: No module named 'libero'`**
A: `libero` 包未正确安装，参考[依赖安装](#依赖安装)中的 LIBERO 安装步骤。

**Q: 数据加载很慢**
A: 首次运行需要解析 TFRecord，之后会自动使用 `.npz` 缓存。若缓存目录已存在且非空，直接从缓存加载。

**Q: CUDA Out of Memory**
A: 尝试减小 `batch_size`（如从 8 改为 4），或减小 `num_queries`（如从 75 改为 50）。

**Q: WandB 登录失败**
A: 在无网络环境下运行 `wandb offline`，或设置环境变量 `WANDB_MODE=offline`。

**Q: rollout 评估时画面渲染失败**
A: 确认服务器支持 offscreen rendering（EGL 或虚拟显示），LIBERO 使用 `OffScreenRenderEnv`，不需要可视化显示器，但需要 OpenGL 支持。可尝试：
```bash
export MUJOCO_GL=egl
python runact_libero.py
```
