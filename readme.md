# ModelScope和HggingFace开源模型的开箱即用
## 本项目的开发初衷在于应用ModelScope和Hugging Face开源模型，一则提供开源模型的使用样例，二是提供开箱即用的工具。当前，本人只是进行了初步的应用，展示了一些demo。
### 一些小问题：
- Q：为什么选择ModelScope和Hugging Face?
- A：ModelScope存在大量的适用于中文任务的深度学习模型，而Hugging Face则适用于中文以外的任务。幸运的是，ModelScope和Hugging Face的框架很相似，可以很好地在这两者之间进行代码的迁移。此外，感兴趣的开发者也可以简单地替换深度学习模型来进行自己任务的开发。

- Q：怎样使用开箱即用的工具？
- A：首先，你需要创建一个python虚拟环境，其次使用根据 *requirements.txt* 安装依赖库，最后使用“python xxx”的形式调用python脚本。