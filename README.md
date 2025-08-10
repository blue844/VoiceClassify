# 语音性别分类项目

这个项目是一个基于深度学习的语音性别分类器，可以自动识别音频文件中说话者的性别（男性/女性）。当然，也可以用于其他简单的语音分类项目。

## 项目结构

```
VoiceClassify/
├── LICENSE
├── README.md
├── datasets/
│   ├── test-00002-of-00003.parquet
│   └── train-00000-of-00009.parquet
├── inference.py
├── inference_gender_audio.py
├── organized_audio/
├── outputs/
│   ├── best_model.pt
│   └── label_map.json
└── train.py
```

## 功能介绍

1. **模型训练** (`train.py`):
   - 支持从指定目录加载音频数据
   - 可配置音频特征提取参数（MFCC 或 Log-Mel 特征）
   - 支持 GPU 加速训练
   - 自动保存最佳模型

2. **模型推理** (`inference_gender_audio.py`):
   - 支持单文件或批量目录推理
   - 输出预测标签、概率及 Top-K 结果
   - 支持 GPU 加速推理

## 安装依赖

```bash
# 创建虚拟环境
conda create -n voice-classify python=3.8
conda activate voice-classify

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy librosa soundfile scikit-learn tqdm
```

## 使用方法

### 1. 训练模型

1. 准备训练数据，将音频文件组织如下：
   ```
   organized_audio/
   ├── male/
   │   └── *.wav
   └── female/
       └── *.wav
   ```

2. 修改 `train.py` 中的配置参数：
   ```python
   cfg = Config(
       DATA_DIRS={
           "male": [r"./organized_audio/male"],
           "female": [r"./organized_audio/female"]
       },
       epochs=30,
       learning_rate=1e-3,
       use_gpu=True,
       batch_size=32,
   )
   ```

3. 运行训练脚本：
   ```bash
   python train.py
   ```

### 2. 模型推理

1. 准备训练好的模型和标签映射文件（位于 `outputs/` 目录）

2. 修改 `inference_gender_audio.py` 中的路径设置：
   ```python
   if __name__ == "__main__":
       # 你可以直接在这里修改为自己的路径
       MODEL_PATH = "./outputs/best_model.pt"
       LABEL_MAP_PATH = "./outputs/label_map.json"

       # 单文件或目录（目录将批量推理 .wav）
       INPUT = "./organized_audio/test/male"  # 例如: "./to_infer" 或 "./some.wav"

       run_inference(INPUT, MODEL_PATH, LABEL_MAP_PATH, cfg)
   ```

3. 运行推理脚本：
   ```bash
   python inference_gender_audio.py
   ```

## 配置参数

### 训练配置 (`train.py`)

- `DATA_DIRS`: 训练数据目录映射
- `sample_rate`: 音频采样率
- `target_duration_sec`: 音频统一长度（秒）
- `use_mfcc`: 是否使用 MFCC 特征（False 表示使用 Log-Mel 特征）
- `n_mels`: Mel 滤波器数量
- `n_mfcc`: MFCC 特征数量
- `n_fft`: FFT 窗口大小
- `hop_length`: 帧移
- `val_size`: 验证集比例
- `batch_size`: 批次大小
- `epochs`: 训练轮数
- `learning_rate`: 学习率
- `use_gpu`: 是否使用 GPU

### 推理配置 (`inference_gender_audio.py`)

- `sample_rate`: 音频采样率
- `target_duration_sec`: 音频统一长度（秒）
- `use_mfcc`: 是否使用 MFCC 特征（需与训练时一致）
- `top_k`: 返回的 top-k 预测结果数量
- `recursive`: 目录推理时是否递归子目录
- `use_gpu`: 是否使用 GPU

## 模型架构

项目使用了一个简单但有效的 CNN 模型，包含以下部分：
1. 卷积块（包含卷积、批归一化、ReLU 激活和池化）
2. 自适应平均池化层
3. 全连接分类器（包含 dropout 正则化）

## 注意事项
1. 确保训练和推理时的特征提取参数一致
2. 支持的音频格式：wav, mp3, flac, m4a, ogg
3. 为获得最佳性能，建议使用 GPU 进行训练和推理

## 许可证
本项目使用 [LICENSE](LICENSE) 许可证。
