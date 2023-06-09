# ModelScope和HggingFace开源模型的开箱即用
## 本项目的开发初衷在于应用[ModelScope](https://www.modelscope.cn/home)和[Hugging Face](https://huggingface.co/)开源模型，一则提供开源模型的使用样例，二是提供开箱即用的工具。当前，本人只是进行了初步的应用，展示了一些demo。



### 一些小问题：
- Q：为什么选择ModelScope和Hugging Face?
- A：ModelScope存在大量的适用于中文任务的深度学习模型，而Hugging Face则适用于中文以外的任务。幸运的是，ModelScope和Hugging Face的框架很相似，可以很好地在这两者之间进行代码的迁移。此外，感兴趣的开发者也可以简单地替换深度学习模型来进行自己任务的开发。

- Q：怎样使用开箱即用的工具？
- A：首先，你需要创建一个python虚拟环境，其次使用根据 *requirements.txt* 安装依赖库，最后使用“python xxx”的形式调用python脚本。

***
### 推荐平台配置***
- Linux：Ubuntu
- GPU：Nvidia

***
### 开箱即用的小工具列表
脚本|简要功能描述|
:-:|:-:|
auto_subtitle.py|自动添加字幕到视频

- auto_subtitle.py：一个可以自动添加字幕的脚本。用户可以指定视频文件、语言、字幕大小、字幕长度等参数来生成带字幕的视频。当前，该脚本仅支持中文和英文字幕。更多的语言待我后续有空添加。也各位欢迎讨论更强大的模型。具体功能和使用模型如下所示：

语言|模型|
:-:|:-:|
中文|damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
英文|openai/whisper-large-v2

***
### 用法
- 创建虚拟环境和安装依赖库。其中，创建虚拟环境采用的是Anaconda或者Miniconda软件。为了减少库冲突，建议统一使用pip安装。
```
conda create -n OutofBox python=3.8
conda activate OutofBox
pip install -r requirements.txt
```
- 使用脚本
```
    python auto_subtitle.py --video xxx
```