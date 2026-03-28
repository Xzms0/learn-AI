# Task 5 - Application 2 高级框架

> 预计耗时：60 天

## 学习目的

## 作业

### 文档 1 - 现代 LLM

阅读 Hugging Face 官方文档教学，了解现代的 LLM 架构（例如 Pipeline），了解 Transformer 的基本原理，了解微调的基本方法，了解 Hugging Face Hub 的使用方法。

在完成上述任务后，你应该写一份文档来阐述你的理解。

### 作业 1 - 复现 MiniMind

复现 [MiniMind](https://github.com/jingyaogong/minimind)

### 作业 2 - 虚拟伴侣

在本轮任务中，你将会实现一个虚拟伴侣的原型机，至少集成了上下文管理，RAG，Web Search 和 Program 四个功能模块。你可以选择使用 LangChain 或 Dify 来实现这个系统。

如果你有至少一张 3090 的显卡，你可以尝试在本地部署 14B 的模型，否则请调用 API 来作为你的核心模型。

#### LangChain

Langchain 是一个重型的框架，大致是转接头的功能，可以将各种 LLM（包括 Hugging Face 的模型，OpenAI 等）与各种工具（例如搜索引擎、数据库、编程环境等）连接起来，构建复杂的应用。

#### Dify

Dify 是一个轻量级的框架，专注于构建基于 LLM 的应用，提供了简单易用的接口来集成各种功能模块。

#### 上下文管理

上下文管理是指在与用户的交互过程中，模型能够记住之前的对话内容，并根据这些内容来生成更相关和连贯的回答。这对于构建一个虚拟伴侣来说非常重要，因为它需要能够理解用户的需求和情感，并提供个性化的回应。

#### RAG

将剧情文本输入到 RAG 模块中，生成相关的知识片段，并将这些片段作为上下文输入到 LLM 中，以增强模型的回答能力。

#### Web Search

通过集成 Web Search 功能，模型可以实时获取互联网信息，提升回答的准确性和时效性。

例如可以特定获取游戏 Wiki 的内容。

#### Program

在完成一些任务时，模型可能需要执行一些代码来获取结果，例如计算、数据处理等。你可以使用 LangChain 的工具功能来实现这一点，或者使用 Dify 的相关功能。

#### 微调

基于角色的语料进行微调，提升模型在特定角色上的表现。

#### FastAPI / Flask

你应该使用 FastAPI 或 Flask 来搭建一个简单的 Web 服务接口，使得用户可以通过 HTTP 请求与虚拟伴侣进行交互。

#### GR / Streamlit

你可以使用 GR 或 Streamlit 来构建一个简单的用户界面，使得用户可以更方便地与虚拟伴侣进行交互。

## 推荐教程与参考资料

### Hugging Face 相关教程

1. [Hugging Face Transformers 官方文档](https://huggingface.co/docs/transformers/index)
2. [Hugging Face Course (推荐从 NLP 模块开始)](https://huggingface.co/course/chapter1/1)

### LangChain 相关教程

1. [LangChain Python Docs](https://python.langchain.com/docs/get_started/introduction)
2. [LangChain Expressions Language (LCEL) 教程](https://python.langchain.com/docs/expression_language/)
