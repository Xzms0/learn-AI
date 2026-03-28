# Research 2 - 训练自己的模型

> 预计耗时：30 天

## 学习目的

在 Research 1 中，你已经学习了深度学习的基础理论，并初步理解了神经网络、反向传播以及 Transformer 等核心概念。

但如果你真的想做科研，或者未来想进入大模型相关的方向，只理解模型原理还远远不够。

你还需要知道，一个大语言模型究竟是如何从数据出发、被一步步训练出来的。

从预训练，到监督微调，再到后续的偏好对齐，现代大模型的训练流程本质上是一套完整的工程系统，而不只是几行 `model.fit()`。

很多同学在学习人工智能时，会停留在会调用 API 或者会跑别人的推理代码这个阶段；而科研更进一步的能力，在于敢于自己读代码、配环境、改参数、跑训练，并分析模型为什么会得到现在的结果。

因此，Research 2 的核心目标，就是让你第一次真正接触训练自己的模型这件事。

通过 MiniMind 这样的入门项目，你将第一次把 Tokenizer、Pretrain、SFT 等流程串起来，理解一个语言模型从只会续写文本到能够按照指令对话，中间到底发生了什么。

学完 Research 2 后，恭喜你，你将正式迈过会用模型和会训练模型之间的那道门槛。

> 从这里开始，你才算真正进入了大模型工程与科研的世界。

## 学习内容

## 作业

### 作业 2 - 自定义数据集构建与数据增强

CS231n 和很多教程都会直接使用 `torchvision.datasets.CIFAR10`，数据也通常是打包好的。

但现实世界的 AI 工程师，90% 的时间在处理乱七八糟的数据。

本轮作业要求你自己构建一个图像分类数据集，并实现数据增强管道。

#### 拟定题目

请确定一个你感兴趣的多分类主题（至少 3 个类别）。

例如：

1. 猫咪 vs 狗狗 vs 仓鼠
2. 剪刀 vs 石头 vs 布
3. 你最喜欢的三个动漫角色
4. 三种不同的食物
5. 自行拟题目

#### 收集数据

利用爬虫或者手动收集所需要的图片，每个类别至少 50 张图片，并将它们整理到合适的文件夹结构中，例如：

```tree
dataset/
    train/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
        class3/
            img1.jpg
            img2.jpg
            ...
    val/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
        class3/
            img1.jpg
            img2.jpg
            ...
```

要求：

1. 图片不需要裁剪整齐，尺寸可以不一致。
2. 严禁直接下载 Kaggle 或其他网站上已经打包好的 .zip 格式数据集。

#### 数据集类

你需要继承 `torch.utils.data.Dataset` 类，实现你自己的数据集加载器。

你的代码结构应该包含：

* `__init__`：初始化文件路径，读取图片列表。
* `__len__`：返回数据集的大小。
* `__getitem__`：根据索引读取图片，转换色彩空间（如 RGB），并应用数据预处理/增强。

你的数据集中可能包含损坏的图片（无法读取）或者非 RGB 格式（如 CMYK 或灰度图）的图片。你的代码需要具备鲁棒性，能够处理或过滤掉这些异常情况，而不是直接报错崩溃。

#### 数据增强

请使用 torchvision.transforms 或 albumentations 库设计一套增强策略，包括但不限于：

* 几何变换：随机旋转、翻转、裁剪。
* 色彩变换：亮度、对比度、饱和度调整。
* 进阶（选做）：尝试使用 Cutout、Mixup 等高级增强技巧。

#### 验证与可视化

你需要编写脚本来验证你的数据集是否工作正常。

使用 DataLoader 加载一个 Batch 的数据。

利用 matplotlib 将这个 Batch 中的图片及其对应的标签（Label）可视化出来。

展示同一张图片在应用了不同的数据增强策略后的效果对比。

#### 文档要求

在完成作业后，你需要写一份**文档**，内容包括但不限于：

1. 你构建了什么数据集？数据来源是哪里？包含多少个类别和样本？
2. 展示你的 CustomDataset 类的代码，并解释你是如何处理图片读取和异常值的。
3. 展示你的 transforms 配置。
    1. 贴出 DataLoader 读取出的 Batch 可视化截图。
    2. 对比原始图片与增强后的图片，分析增强策略是否合理。
4. 在处理脏数据的过程中，你踩了哪些坑？（例如图片后缀是 .jpg 但实际是 png 格式？图片损坏导致训练中断？）

### 作业 3 - 根据 MNIST 数据集实现一个简单的识别数字的 Demo（选做）

人工智能的高级框架不只有 PyTorch，还有 TensorFlow，Keras 等等。

目前，Keras 已经被集成到 TensorFlow 中，成为其高级 API。

TensorFlow.js 是 TensorFlow 的 JavaScript 版本，可以直接在浏览器中运行深度学习模型。

本次作业要求你实现并部署一个基于 TensorFlow.js 的 MNIST 手写数字识别 Web 应用。

该项目的最终成果可以在 [这里](https://shaddocknh3.github.io/tfjs-mnist-digit-recognizer/) 体验。

本次作业的目标是让你亲历一个人工智能应用从训练到部署的全过程。你可以以 [该项目](https://github.com/ShaddockNH3/tfjs-mnist-digit-recognizer) 为基础框架，但识别模型需要你亲手训练。

> 此后简称为该项目，不列出链接。

#### 训练识别模型

原项目（旧版本）中的模型存在版本兼容性问题，且无法兼容手机端。

由于浏览器环境主要运行 TensorFlow.js，无法直接运行 PyTorch 和 TensorFlow。

你的任务是使用 Keras / PyTorch，在经典的 MNIST 数据集上，从零开始构建并训练一个卷积神经网络（CNN）。

你需要自行设计网络结构、选择优化器和损失函数，并进行超参数调整，以达到尽可能高的识别准确率。

训练完成后，你将得到一个 Keras / PyTorch 模型文件（通常是 `.h5` 格式或 `.pt` 格式）。

#### 模型转换

你需要通过转换工具，将训练好的模型迁移到 Web 环境中。

如果你采用了 Keras 进行训练，你需要使用 `tensorflowjs_converter` 将 Keras 模型直接转换为 TensorFlow.js 格式。

如果你采用了 PyTorch 进行训练，你需要把训练好的模型导出为 ONNX 格式，然后使用 onnx-tf 等工具将 ONNX 模型转换为 TensorFlow SavedModel，再转换为 TensorFlow.js 格式。

最终，你将得到两个文件：`model.json` 和一个 `.bin` 权重文件。

#### 前端实现

你可以 fork 该项目，然后替换原仓库 model 文件夹下的两个模型文件。

在你完成文件替换后，你还需要将 index.html 的第 6 行修改为你自己的 GitHub 用户名。

当然你也可以从零开始实现一个前端页面，要求功能与该项目类似即可。

在你完成整个项目后，你需要通过 GitHub Pages 将项目部署为一个公开可访问的网页。

在 PC 和移动端打开你的网页，在画板上写下数字，验证识别功能是否正常工作。

## 推荐教程与参考资料

### 识别数字 Demo 参考资料

* [参考实现：使用 Keras.js 的旧版思路](https://github.com/starkwang/keras-js-demo)
* [参考教程：从训练到部署的详细步骤](https://www.cnblogs.com/chinasoft/p/17084356.html)
* [TensorFlow.js 官方文档](https://www.tensorflow.org/js/guide?hl=zh-cn)
