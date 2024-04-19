<p align="center">
  <img src="https://github.com/meta-llama/llama3/blob/main/Llama3_Repo.jpeg" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/meta-Llama"> Hugging Face 上的模型</a>&nbsp | <a href="https://ai.meta.com/blog/"> 博客</a>&nbsp |  <a href="https://llama.meta.com/">网站</a>&nbsp | <a href="https://llama.meta.com/get-started/">开始使用</a>&nbsp
<br>

---


# Meta Llama 3

我们正在释放大语言模型的力量。我们最新版本的 Llama 现在可供个人、创作者、研究人员和各种规模的企业使用,以便他们可以负责任地试验、创新和扩展他们的想法。

此版本包括预训练和指令调优的 Llama 3 语言模型的模型权重和起始代码,包括 8B 到 70B 参数的大小。

此存储库旨在作为加载 Llama 3 模型并运行推理的最小示例。有关更详细的示例,请参阅 [llama-recipes](https://github.com/facebookresearch/llama-recipes/)。

## 下载 

为了下载模型权重和分词器,请访问 [Meta Llama 网站](https://llama.meta.com/llama-downloads/)并接受我们的许可协议。

提交请求后,您将通过电子邮件收到一个签名的 URL。然后运行 download.sh 脚本,在提示时传递提供的 URL 以开始下载。

先决条件:确保您已安装 `wget` 和 `md5sum`。然后运行脚本:`./download.sh`。

请记住,链接会在 24 小时和一定次数的下载后过期。如果开始看到 `403: Forbidden` 之类的错误,您始终可以重新请求链接。

### 访问 Hugging Face

我们还在 [Hugging Face](https://huggingface.co/meta-llama) 上提供下载,包括 transformers 和原生 `llama3` 格式。要从 Hugging Face 下载权重,请按照以下步骤操作:

- 访问其中一个仓库,例如 [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)。
- 阅读并接受许可。请求获得批准后,您将获得所有 Llama 3 模型的访问权限。请注意,处理请求通常需要长达一个小时。
- 要下载原始原生权重以与此仓库一起使用,请单击"Files and versions"选项卡,然后下载 `original` 文件夹的内容。如果你安装了 `pip install huggingface-hub`,也可以从命令行下载它们:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --include "original/*" --local-dir meta-llama/Meta-Llama-3-8B-Instruct

- 要与transformers一起使用，以下pipeline代码片段将下载并缓存权重:
```python
  import transformers
  import torch
  
  model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
  
  pipeline = transformers.pipeline(
    "text-generation", 
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
  )

## 快速开始

您可以按照以下步骤快速开始使用 Llama 3 模型。这些步骤将让您能够在本地进行快速推理。更多示例，请查看[Llama 配方仓库](https://github.com/facebookresearch/llama-recipes)。

1. 在一个已安装 PyTorch / CUDA 的 conda 环境中克隆并下载此仓库。

2. 在顶级目录运行：
    ```bash
    pip install -e .
    ```
3. 访问[Meta Llama 网站](https://llama.meta.com/llama-downloads/)并注册以下载模型。

4. 注册后，您将收到一封带有下载模型 URL 的电子邮件。在运行 download.sh 脚本时，您将需要此 URL。

5. 收到电子邮件后，导航至您下载的 llama 仓库并运行 download.sh 脚本。
    - 确保授予 download.sh 脚本执行权限
    - 在此过程中，系统会提示您输入电子邮件中的 URL。
    - 不要使用“复制链接”选项，而是确保手动从电子邮件复制链接。

6. 下载所需的模型后，您可以使用以下命令在本地运行模型：
```bash
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 6

**注意**
- 将 `Meta-Llama-3-8B-Instruct/` 替换为您的检查点目录路径，将 `Meta-Llama-3-8B-Instruct/tokenizer.model` 替换为您的分词器模型路径。
- `–nproc_per_node` 应设置为您使用的模型的 [MP](#inference) 值。
- 根据需要调整 `max_seq_len` 和 `max_batch_size` 参数。
- 此示例运行此仓库中找到的 [example_chat_completion.py](example_chat_completion.py)，但您可以更改为不同的 .py 文件。

## 推理

不同的模型需要不同的模型并行（MP）值：

|  模型  | MP |
|--------|----|
| 8B     | 1  |
| 70B    | 8  |

所有模型支持最多 8192 令牌的序列长度，但我们会根据 `max_seq_len` 和 `max_batch_size` 的值预分配缓存。因此，请根据您的硬件设置这些值。

### 预训练模型

这些模型未针对聊天或问答进行微调。应该设定提示，使得预期答案是提示的自然延续。

请参见 `example_text_completion.py` 以获取一些示例。为了说明，参见下面的命令，以使用 llama-3-8b 模型运行它（`nproc_per_node` 需要设置为 `MP` 值）：

torchrun --nproc_per_node 1 example_text_completion.py
--ckpt_dir Meta-Llama-3-8B/
--tokenizer_path Meta-Llama-3-8B/tokenizer.model
--max_seq_len 128 --max_batch_size 4


### 指令调整模型

微调模型是为对话应用培训的。为了获得它们的预期特性和性能，需要遵循 [`ChatFormat`](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L202) 中定义的特定格式：提示以特殊令牌 `<|begin_of_text|>` 开始，之后跟随一个或多个消息。每条消息以标签 `<|start_header_id|>` 开始，角色为 `system`、`user` 或 `assistant`，并以标签 `<|end_header_id|>` 结束。在双换行 `\n\n` 之后，消息的内容随之而来。每条消息的结尾由 `<|eot_id|>` 令牌标记。

您还可以部署额外的分类器，以过滤掉被认为不安全的输入和输出。请参见 llama-recipes 仓库中的 [一个示例](https://github.com/meta-llama/llama-recipes/blob/main/recipes/inference/local_inference/inference.py)，了解如何在您的推理代码的输入和输出中添加安全检查器。

使用 llama-3-8b-chat 的示例：

torchrun --nproc_per_node 1 example_chat_completion.py
--ckpt_dir Meta-Llama-3-8B-Instruct/
--tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model
--max_seq_len 512 --max_batch_size 6



Llama 3 是一项新技术，使用时带有潜在风险。迄今为止进行的测试没有——也不可能——覆盖所有情况。
为了帮助开发者应对这些风险，我们创建了 [负责任使用指南](https://ai.meta.com/static-resource/responsible-use-guide/)。

## 问题

请通过以下方式之一报告软件“bug”或模型的其他问题：
- 报告模型问题：[https://github.com/meta-llama/llama3/issues](https://github.com/meta-llama/llama3/issues)
- 报告模型生成的风险内容：[developers.facebook.com/llama_output_feedback](http://developers.facebook.com/llama_output_feedback)
- 报告漏洞和安全问题：[facebook.com/whitehat/info](http://facebook.com/whitehat/info)

## 模型卡片
参见 [MODEL_CARD.md](MODEL_CARD.md)。

## 许可证

我们的模型和权重为研究者和商业实体授权，坚持开放原则。我们的使命是通过这一机会赋能个人和行业，同时促进发现和道德 AI 进步的环境。

请查看 [LICENSE](LICENSE) 文件，以及我们的 [可接受使用政策](USE_POLICY.md)

## 问题

对于常见问题，可以在此处找到 FAQ [https://llama.meta.com/faq](https://llama.meta.com/faq)，随着新问题的出现，这将不断更新。