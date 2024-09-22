
<div align="center">

<img src="docs/imgs/Title.jpg" />

# NanoDet-Plus

**Super fast and high accuracy lightweight anchor-free object detection model. Real-time on mobile devices.**

[![CI testing](https://img.shields.io/github/checks-status/RangiLyu/nanodet/main?label=CI&style=flat)](https://img.shields.io/github/checks-status/RangiLyu/nanodet/main?label=CI&style=flat) ![Codecov](https://img.shields.io/codecov/c/github/RangiLyu/nanodet?color=hotpink) [![GitHub license](https://img.shields.io/github/license/RangiLyu/nanodet?color=turquoise&style=flat)](https://github.com/RangiLyu/nanodet/blob/main/LICENSE) [![Github downloads](https://img.shields.io/github/downloads/RangiLyu/nanodet/total?color=orange&label=downloads&logo=github&logoColor=lightgrey&style=flat)](https://img.shields.io/github/downloads/RangiLyu/nanodet/total?color=yellow&label=Downloads&logo=github&logoColor=lightgrey&style=flat) [![GitHub release (latest by date)](https://img.shields.io/github/v/release/RangiLyu/nanodet?style=flat)](https://img.shields.io/github/v/release/RangiLyu/nanodet?style=flat)

</div>

## 1. 简介

NanoDet 是最轻量的目标检测模型之一（还有 FastestDet），实测 NanoDet-m-0.5x 分辨率为 256×160 时可以在 J1900 纯 cpu 上跑到 28fps。

中文文档中主要内容如下：

- 训练模型（配置环境、准备数据、修改参数）
- 部署模型（模型转换、配置环境、运行程序）

## 2. 训练模型

### 2.1.  配置 python 环境

#### 2.1.1. 创建 conda 环境

```sh
conda create -n nanodet python=3.8 -y
conda activate nanodet
```

#### 2.1.2. 安装 pytorch

最新版本安装命令：<https://pytorch.org/get-started/locally/>
历史版本安装命令：<https://pytorch.org/get-started/previous-versions/>

#### 2.1.3. 安装其他 pip 库

```sh
pip install -r requirements.txt
python setup.py develop
```

### 2.2. 准备训练数据

### 2.2.1. 准备数据文件

准备数据文件夹，内容如下：

```txt
data
├─ train
│  ├─ 001.png
│  └─ 002.png
├─ val
│  ├─ 003.png
│  └─ 004.png
└ classes.txt
```

其中 `classes.txt` 内容如下：

```txt
class_1
class_2
```

#### 2.2.2. 安装标注软件

傻瓜式标注软件 X-AnyLabeling：<https://github.com/CVHub520/X-AnyLabeling/releases>
软件文档：<https://github.com/CVHub520/X-AnyLabeling/blob/main/README_zh-CN.md>

#### 2.2.3. 标注数据

1. 标注：打开标注软件 X-AnyLabeling，打开 `train` 或 `val` 文件夹，开始标注文件夹内图片，即绘制矩形框、设置矩形框的类别标签。
2. 生成单一标注文件：一个文件夹内图片标注完成后，点击 `格式` 菜单并选择 `COCO`，随后在弹出窗口中选择 `classes.txt` 文件，即可在图片文件夹中生成 `annotations/instances_default.json` 文件供训练网络用。
3. 删除每张图片单独的标注文件，最后数据文件夹内容如下：

```txt
data
├─ train
|  ├─ annotations
|  |  └─ instances_default.json
|  ├─ 001.png
│  └─ 002.png
├─ val
|  ├─ annotations
|  |  └─ instances_default.json
│  ├─ 003.png
│  └─ 004.png
└ classes.txt
```

### 2.3. 修改参数

基于 [config/temple.yml](config/temple.yml) 修改所需的参数。

#### 2.3.1. 修改模型参数

根据选用的模型，使用该模型参数模板中的 `model` 类替换 `temple.yml` 中的 `model` 类：

Model | config |
:-:|:-:|
NanoDet-m-0.5x | [config/models/nanodet-m-0.5x.yml](config/models/nanodet-m-0.5x.yml) |
NanoDet-m-1.0x | [config/models/nanodet-m-1.0x.yml](config/models/nanodet-m-1.0x.yml) |
NanoDet-m-1.5x | [config/models/nanodet-m-1.5x.yml](config/models/nanodet-m-1.5x.yml) |
NanoDet-plus-m-1.0x | [config/models/nanodet-plus-m-1.0x.yml](config/models/nanodet-plus-m-1.0x.yml) |
NanoDet-plus-m-1.5x | [config/models/nanodet-plus-m-1.5x.yml](config/models/nanodet-plus-m-1.5x.yml) |

- 模型越小，推理越快，但精度越低。

#### 2.3.2. 修改必须的数据参数

```yml
model:
  head:
    num_classes: 2  # 类别数

class_names: ['class_name_1', 'class_name_2']  # 类名

data:
  train:
    img_path: /home/usr/data/train  # 训练数据所在的路径
    ann_path: /home/usr/data/train/annotations/instances_default.json  # 训练数据标注文件路径
    input_size: [320,320]  # 数据训练分辨率，[w,h]
  val:
    img_path: /home/usr/data/val  # 训练数据所在的路径
    ann_path: /home/usr/data/val/annotations/instances_default.json  # 训练数据标注文件路径
    input_size: [320,320]  # 数据推理分辨率，[w,h]
```

- 数据分辨率越小，模型推理越快，但精度越低。
- 可以设置非 1:1 的分辨率。
- 分辨率长、宽均需是 32 的倍数。

#### 2.3.3. 修改必须的训练参数

```yml
device:
  gpu_ids: [0,1]  # 使用 gpu 训练
  # gpu_ids: -1  # 使用 cpu 训练
  batchsize_per_gpu: 96  # batch size
```

### 2.4. 开始训练

```sh
python tools/train.py [config_file_path]
```

在配置文件设置的 `save_dir` 路径中运行 `tensorboard` 即可在浏览器中（<http://localhost:6006>）观察训练状态：

```sh
cd [save_dir]
tensorboard --logdir .
```

### 2.5. 测试模型推理效果

使用以下命令测试 pytorch 模型（model_best.pth）的推理效果：

```sh
# 推理单张图片
python demo/demo.py image --config [CONFIG_PATH] --model [MODEL_PATH] --path [IMAGE_PATH]

# 推理视频文件
python demo/demo.py video --config [CONFIG_PATH] --model [MODEL_PATH] --path [VIDEO_PATH]
```

## 3. 部署模型

### 3.1. 模型转换

1、将 pytorch 模型转为 onnx 模型：

运行以下命令后会在 save_dir 中具体的实验目录下生成 `nanodet.onnx` 模型文件：

```sh
python tools/export_onnx.py --path [save_dir/log_dir]
```

2、将 onnx 模型转为 ncnn 模型：

使用 `pnnx` 转换：<https://github.com/pnnx/pnnx>

```sh
python -m pip install pnnx
# pnnx [model_path] inputshape=[input_shape]
pnnx nanodet.onnx inputshape=[1,3,416,416]
```

### 3.2. 配置部署环境

1、安装 opencv。

2、安装 `ncnn`：<https://github.com/Tencent/ncnn/releases>
下载并解压 `ncnn-full-source.zip`

```sh
unzip ncnn-xxx-full-source.zip
cd ncnn-xxx-full-source
mkdir build && cd build
cmake ..
make
sudo make install
```

### 3.3. 编译部署代码

部署代码为 `demo_ncnn` 文件夹。

编译前需先：

1、修改 `CMakeLists.txt` 中 `SET(ncnn_DIR /home/usr/ncnn/build/install/lib/cmake/ncnn)` 为实际 ncnn 安装路径。
2、修改 `main.cpp` 中的 `class_names` 和 `nanodet.h` 中的 `labels` 为具体类别名。

```sh
cd demo_ncnn
mkdir build && cd build
cmake ..
make
```

### 3.4. 运行部署程序

首先将  `xxx.ncnn.bin` 和  `xxx.ncnn.param` 改名为  `nanodet.bin` 和  `nanodet.param` 并放置在 `demo_ncnn` 文件夹下。

```sh
cd demo_ncnn/build

# 测试推理速度
./nanodet_demo 3 0

# 推理单张图片
./nanodet_demo 1 [img_path]

# 推理视频文件
./nanodet_demo 2 [video_path]
```

****

- ⚡Super lightweight: Model file is only 980KB(INT8) or 1.8MB(FP16).
- ⚡Super fast: 97fps(10.23ms) on mobile ARM CPU.
- 👍High accuracy: Up to **34.3 mAP<sup>val</sup>@0.5:0.95** and still realtime on CPU.
- 🤗Training friendly:  Much lower GPU memory cost than other models. Batch-size=80 is available on GTX1060 6G.
- 😎Easy to deploy: Support various backends including **ncnn, MNN and OpenVINO**. Also provide **Android demo** based on ncnn inference framework.

## Introduction

![](docs/imgs/nanodet-plus-arch.png)

NanoDet is a FCOS-style one-stage anchor-free object detection model which using [Generalized Focal Loss](https://arxiv.org/pdf/2006.04388.pdf) as classification and regression loss.

In NanoDet-Plus, we propose a novel label assignment strategy with a simple **assign guidance module (AGM)** and a **dynamic soft label assigner (DSLA)** to solve the optimal label assignment problem in lightweight model training. We also introduce a light feature pyramid called Ghost-PAN to enhance multi-layer feature fusion. These improvements boost previous NanoDet's detection accuracy by **7 mAP** on COCO dataset.

[NanoDet-Plus 知乎中文介绍](https://zhuanlan.zhihu.com/p/449912627)

[NanoDet 知乎中文介绍](https://zhuanlan.zhihu.com/p/306530300)

QQ交流群：908606542 (答案：炼丹)

****

## Benchmarks

Model          |Resolution| mAP<sup>val<br>0.5:0.95 |CPU Latency<sup><br>(i7-8700) |ARM Latency<sup><br>(4xA76) | FLOPS      |   Params  | Model Size
:-------------:|:--------:|:-------:|:--------------------:|:--------------------:|:----------:|:---------:|:-------:
NanoDet-m      | 320*320 |   20.6   | **4.98ms**           | **10.23ms**          | **0.72G**  | **0.95M** | **1.8MB(FP16)** &#124; **980KB(INT8)**
**NanoDet-Plus-m** | 320*320 | **27.0** | **5.25ms**       | **11.97ms**          | **0.9G**   | **1.17M** | **2.3MB(FP16)** &#124; **1.2MB(INT8)**
**NanoDet-Plus-m** | 416*416 | **30.4** | **8.32ms**       | **19.77ms**          | **1.52G**  | **1.17M** | **2.3MB(FP16)** &#124; **1.2MB(INT8)**
**NanoDet-Plus-m-1.5x** | 320*320 | **29.9** | **7.21ms**  | **15.90ms**          | **1.75G**  | **2.44M** | **4.7MB(FP16)** &#124; **2.3MB(INT8)**
**NanoDet-Plus-m-1.5x** | 416*416 | **34.1** | **11.50ms** | **25.49ms**          | **2.97G**   | **2.44M** | **4.7MB(FP16)** &#124; **2.3MB(INT8)**
YOLOv3-Tiny    | 416*416 |   16.6   | -                    | 37.6ms               | 5.62G      | 8.86M     |   33.7MB
YOLOv4-Tiny    | 416*416 |   21.7   | -                    | 32.81ms              | 6.96G      | 6.06M     |   23.0MB
YOLOX-Nano     | 416*416 |   25.8   | -                    | 23.08ms              | 1.08G      | 0.91M     |   1.8MB(FP16)
YOLOv5-n       | 640*640 |   28.4   | -                    | 44.39ms              | 4.5G       | 1.9M      |   3.8MB(FP16)
FBNetV5        | 320*640 |   30.4   | -                    | -                    | 1.8G       | -         |   -
MobileDet      | 320*320 |   25.6   | -                    | -                    | 0.9G       | -         |   -

***Download pre-trained models and find more models in [Model Zoo](#model-zoo) or in [Release Files](https://github.com/RangiLyu/nanodet/releases)***

<details>
    <summary>Notes (click to expand)</summary>

- ARM Performance is measured on Kirin 980(4xA76+4xA55) ARM CPU based on ncnn. You can test latency on your phone with [ncnn_android_benchmark](https://github.com/nihui/ncnn-android-benchmark).

- Intel CPU Performance is measured Intel Core-i7-8700 based on OpenVINO.

- NanoDet mAP(0.5:0.95) is validated on COCO val2017 dataset with no testing time augmentation.

- YOLOv3&YOLOv4 mAP refers from [Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036).

</details>

****

## NEWS

- [2023.01.20] Upgrade to [pytorch-lightning-1.9](https://github.com/Lightning-AI/lightning/releases/tag/1.9.0). The minimum PyTorch version is upgraded to 1.10. Support FP16 training(Thanks @crisp-snakey). Support ignore label(Thanks @zero0kiriyu).

- [2022.08.26] Upgrade to [pytorch-lightning-1.7](https://lightning.ai/). The minimum PyTorch version is upgraded to 1.9. To use previous version of PyTorch, please install [NanoDet <= v1.0.0-alpha-1](https://github.com/RangiLyu/nanodet/tags)

- [2021.12.25] **NanoDet-Plus** release! Adding **AGM**(Assign Guidance Module) & **DSLA**(Dynamic Soft Label Assigner) to improve **7 mAP** with only a little cost.

Find more update notes in [Update notes](docs/update.md).

## Demo

### Android demo

![android_demo](docs/imgs/Android_demo.jpg)

Android demo project is in ***demo_android_ncnn*** folder. Please refer to [Android demo guide](demo_android_ncnn/README.md).

Here is a better implementation 👉 [ncnn-android-nanodet](https://github.com/nihui/ncnn-android-nanodet)

### NCNN C++ demo

C++ demo based on [ncnn](https://github.com/Tencent/ncnn) is in ***demo_ncnn*** folder. Please refer to [Cpp demo guide](demo_ncnn/README.md).

### MNN demo

Inference using [Alibaba's MNN framework](https://github.com/alibaba/MNN) is in ***demo_mnn*** folder. Please refer to [MNN demo guide](demo_mnn/README.md).

### OpenVINO demo

Inference using [OpenVINO](https://01.org/openvinotoolkit) is in ***demo_openvino*** folder. Please refer to [OpenVINO demo guide](demo_openvino/README.md).

### Web browser demo

<https://nihui.github.io/ncnn-webassembly-nanodet/>

### Pytorch demo

First, install requirements and setup NanoDet following installation guide. Then download COCO pretrain weight from here

👉[COCO pretrain checkpoint](https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416_checkpoint.ckpt)

The pre-trained weight was trained by the config `config/nanodet-plus-m_416.yml`.

- Inference images

```bash
python demo/demo.py image --config CONFIG_PATH --model MODEL_PATH --path IMAGE_PATH
```

- Inference video

```bash
python demo/demo.py video --config CONFIG_PATH --model MODEL_PATH --path VIDEO_PATH
```

- Inference webcam

```bash
python demo/demo.py webcam --config CONFIG_PATH --model MODEL_PATH --camid YOUR_CAMERA_ID
```

Besides, We provide a notebook [here](./demo/demo-inference-with-pytorch.ipynb) to demonstrate how to make it work with PyTorch.

****

## Install

### Requirements

- Linux or MacOS
- CUDA >= 10.2
- Python >= 3.7
- Pytorch >= 1.10.0, <2.0.0

### Step

1. Create a conda virtual environment and then activate it.

```shell script
conda create -n nanodet python=3.8 -y
conda activate nanodet
```

2. Install pytorch

```shell script
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
```

3. Clone this repository

```shell script
git clone https://github.com/RangiLyu/nanodet.git
cd nanodet
```

4. Install requirements

```shell script
pip install -r requirements.txt
```

5. Setup NanoDet

```shell script
python setup.py develop
```

****

## Model Zoo

NanoDet supports variety of backbones. Go to the [***config*** folder](config/) to see the sample training config files.

Model                 | Backbone           |Resolution|COCO mAP| FLOPS |Params | Pre-train weight |
:--------------------:|:------------------:|:--------:|:------:|:-----:|:-----:|:-----:|
NanoDet-m             | ShuffleNetV2 1.0x  | 320*320  |  20.6  | 0.72G | 0.95M | [Download](https://drive.google.com/file/d/1ZkYucuLusJrCb_i63Lid0kYyyLvEiGN3/view?usp=sharing) |
NanoDet-Plus-m-320 (***NEW***)     | ShuffleNetV2 1.0x | 320*320  |  27.0  | 0.9G  | 1.17M | [Weight](https://drive.google.com/file/d/1Dq0cTIdJDUhQxJe45z6rWncbZmOyh1Tv/view?usp=sharing) &#124; [Checkpoint](https://drive.google.com/file/d/1YvuEhahlgqxIhJu7bsL-fhaqubKcCWQc/view?usp=sharing)
NanoDet-Plus-m-416 (***NEW***)     | ShuffleNetV2 1.0x | 416*416  |  30.4  | 1.52G | 1.17M | [Weight](https://drive.google.com/file/d/1FN3WK3FLjBm7oCqiwUcD3m3MjfqxuzXe/view?usp=sharing) &#124; [Checkpoint](https://drive.google.com/file/d/1gFjyrl7O8p5APr1ZOtWEm3tQNN35zi_W/view?usp=sharing)
NanoDet-Plus-m-1.5x-320 (***NEW***)| ShuffleNetV2 1.5x | 320*320  |  29.9  | 1.75G | 2.44M | [Weight](https://drive.google.com/file/d/1Xdlgu5lxiS3w6ER7GE1mZpY663wmpcyY/view?usp=sharing) &#124; [Checkpoint](https://drive.google.com/file/d/1qXR6t3TBMXlz6GlTU3fxiLA-eueYoGrW/view?usp=sharing)
NanoDet-Plus-m-1.5x-416 (***NEW***)| ShuffleNetV2 1.5x | 416*416  |  34.1  | 2.97G | 2.44M | [Weight](https://drive.google.com/file/d/16FJJJgUt5VrSKG7RM_ImdKKzhJ-Mu45I/view?usp=sharing) &#124; [Checkpoint](https://drive.google.com/file/d/17sdAUydlEXCrHMsxlDPLj5cGb-8-mmY6/view?usp=sharing)

*Notice*: The difference between `Weight` and `Checkpoint` is the weight only provide params in inference time, but the checkpoint contains training time params.

Legacy Model Zoo

Model                 | Backbone           |Resolution|COCO mAP| FLOPS |Params | Pre-train weight |
:--------------------:|:------------------:|:--------:|:------:|:-----:|:-----:|:-----:|
NanoDet-m-416         | ShuffleNetV2 1.0x  | 416*416  |  23.5  |  1.2G | 0.95M | [Download](https://drive.google.com/file/d/1jY-Um2VDDEhuVhluP9lE70rG83eXQYhV/view?usp=sharing)|
NanoDet-m-1.5x        | ShuffleNetV2 1.5x  | 320*320  |  23.5  | 1.44G | 2.08M | [Download](https://drive.google.com/file/d/1_n1cAWy622LV8wbUnXImtcvcUVPOhYrW/view?usp=sharing) |
NanoDet-m-1.5x-416    | ShuffleNetV2 1.5x  | 416*416  |  26.8  | 2.42G | 2.08M | [Download](https://drive.google.com/file/d/1CCYgwX3LWfN7hukcomhEhGWN-qcC3Tv4/view?usp=sharing)|
NanoDet-m-0.5x        | ShuffleNetV2 0.5x  | 320*320  |  13.5  |  0.3G | 0.28M | [Download](https://drive.google.com/file/d/1rMHkD30jacjRpslmQja5jls86xd0YssR/view?usp=sharing) |
NanoDet-t             | ShuffleNetV2 1.0x  | 320*320  |  21.7  | 0.96G | 1.36M | [Download](https://drive.google.com/file/d/1TqRGZeOKVCb98ehTaE0gJEuND6jxwaqN/view?usp=sharing) |
NanoDet-g             | Custom CSP Net     | 416*416  |  22.9  |  4.2G | 3.81M | [Download](https://drive.google.com/file/d/1f2lH7Ae1AY04g20zTZY7JS_dKKP37hvE/view?usp=sharing)|
NanoDet-EfficientLite | EfficientNet-Lite0 | 320*320  |  24.7  | 1.72G | 3.11M | [Download](https://drive.google.com/file/d/1Dj1nBFc78GHDI9Wn8b3X4MTiIV2el8qP/view?usp=sharing)|
NanoDet-EfficientLite | EfficientNet-Lite1 | 416*416  |  30.3  | 4.06G | 4.01M | [Download](https://drive.google.com/file/d/1ernkb_XhnKMPdCBBtUEdwxIIBF6UVnXq/view?usp=sharing) |
NanoDet-EfficientLite | EfficientNet-Lite2 | 512*512  |  32.6  | 7.12G | 4.71M | [Download](https://drive.google.com/file/d/11V20AxXe6bTHyw3aMkgsZVzLOB31seoc/view?usp=sharing) |
NanoDet-RepVGG        | RepVGG-A0          | 416*416  |  27.8  | 11.3G | 6.75M | [Download](https://drive.google.com/file/d/1nWZZ1qXb1HuIXwPSYpEyFHHqX05GaFer/view?usp=sharing) |

****

## How to Train

1. **Prepare dataset**

    If your dataset annotations are pascal voc xml format, refer to [config/nanodet_custom_xml_dataset.yml](config/nanodet_custom_xml_dataset.yml)

    Otherwise, if your dataset annotations are YOLO format ([Darknet TXT](https://github.com/AlexeyAB/Yolo_mark/issues/60#issuecomment-401854885)), refer to [config/nanodet-plus-m_416-yolo.yml](config/nanodet-plus-m_416-yolo.yml)

    Or convert your dataset annotations to MS COCO format[(COCO annotation format details)](https://cocodataset.org/#format-data).

2. **Prepare config file**

    Copy and modify an example yml config file in config/ folder.

    Change ***save_dir*** to where you want to save model.

    Change ***num_classes*** in ***model->arch->head***.

    Change image path and annotation path in both ***data->train*** and ***data->val***.

    Set gpu ids, num workers and batch size in ***device*** to fit your device.

    Set ***total_epochs***, ***lr*** and ***lr_schedule*** according to your dataset and batchsize.

    If you want to modify network, data augmentation or other things, please refer to [Config File Detail](docs/config_file_detail.md)

3. **Start training**

   NanoDet is now using [pytorch lightning](https://github.com/PyTorchLightning/pytorch-lightning) for training.

   For both single-GPU or multiple-GPUs, run:

   ```shell script
   python tools/train.py CONFIG_FILE_PATH
   ```

4. **Visualize Logs**

    TensorBoard logs are saved in `save_dir` which you set in config file.

    To visualize tensorboard logs, run:

    ```shell script
    cd <YOUR_SAVE_DIR>
    tensorboard --logdir ./
    ```

****

## How to Deploy

NanoDet provide multi-backend C++ demo including ncnn, OpenVINO and MNN.
There is also an Android demo based on ncnn library.

### Export model to ONNX

To convert NanoDet pytorch model to ncnn, you can choose this way: pytorch->onnx->ncnn

To export onnx model, run `tools/export_onnx.py`.

```shell script
python tools/export_onnx.py --cfg_path ${CONFIG_PATH} --model_path ${PYTORCH_MODEL_PATH}
```

### Run NanoDet in C++ with inference libraries

### ncnn

Please refer to [demo_ncnn](demo_ncnn/README.md).

### OpenVINO

Please refer to [demo_openvino](demo_openvino/README.md).

### MNN

Please refer to [demo_mnn](demo_mnn/README.md).

### Run NanoDet on Android

Please refer to [android_demo](demo_android_ncnn/README.md).

****

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{=nanodet,
    title={NanoDet-Plus: Super fast and high accuracy lightweight anchor-free object detection model.},
    author={RangiLyu},
    howpublished = {\url{https://github.com/RangiLyu/nanodet}},
    year={2021}
}
```

****

## Thanks

<https://github.com/Tencent/ncnn>

<https://github.com/open-mmlab/mmdetection>

<https://github.com/implus/GFocal>

<https://github.com/cmdbug/YOLOv5_NCNN>

<https://github.com/rbgirshick/yacs>
