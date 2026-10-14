# MokioMind 源码对照实现

实现依据：[Wood-Q/MokioMind](https://github.com/Wood-Q/MokioMind)，
固定版本为 [`125dc2d245ee35f4940855efabb915a0ad02428c`](https://github.com/Wood-Q/MokioMind/tree/125dc2d245ee35f4940855efabb915a0ad02428c)。

## 本地文件与原项目的对应关系

| 本地文件 | 原项目文件 | 差异 |
| --- | --- | --- |
| `model/model.py` | `model/MokioModel.py` | 内容一致 |
| `dataset/Im_dataset.py` | `dataset/lm_dataset.py` | 内容一致 |
| `trainer/trian_pretrain.py` | `trainer/train_pretrain.py` | 仅适配模型、数据集的本地导入路径 |
| `trainer/trainer_utils.py` | `trainer/trainer_utils.py` | 仅适配模型的本地导入路径 |
| `model/tokenizer.json` | `model/tokenizer.json` | 内容一致 |
| `model/tokenizer_config.json` | `model/tokenizer_config.json` | 内容一致 |

保留原项目的 Hugging Face `load_dataset`、元组形式的批次、loss/aux_loss 输出、
GQA、RoPE、梯度累积和检查点逻辑。普通模型使用 `--use_moe 0`。
依赖配置补充源码实际使用的 `datasets`；Python 版本保持本地的 3.12。

## 验证普通模型

在项目根目录运行：

```bash
.venv/bin/python trainer/smoke_pretrain.py
.venv/bin/python -m unittest discover -s tests -v
```

验证脚本使用原项目的分词器和训练入口，在 CPU 上训练小模型，临时数据和检查点会在结束后清理。
测试和验证脚本只用于检查运行结果，不更改原项目的模型或训练算法。

## 使用自己的预训练数据

准备 UTF-8 JSONL 文件，每行是一条文本：

```json
{"text": "这是一条预训练文本。"}
{"text": "这是另一条文本。"}
```

源码中的默认路径以 `trainer` 目录为工作目录：

```bash
cd trainer
../.venv/bin/python trian_pretrain.py \
  --data_path /absolute/path/train.jsonl \
  --device cpu --num_workers 0 \
  --hidden_size 128 --num_hidden_layers 2 \
  --batch_size 2 --max_seq_len 128 --accumulation_steps 1 --use_moe 0
```

有 CUDA 时可使用 `--device cuda:0`。启用 `--use_wandb` 需要另外安装 `swanlab`。
权重默认保存到 `../out`，续训检查点保存到 `../checkpoints`。

原项目只在步数能被 `accumulation_steps` 整除时更新参数，末尾不足一组的梯度没有单独处理。
本次保留该源码行为；验证时使用 `accumulation_steps=1`。
