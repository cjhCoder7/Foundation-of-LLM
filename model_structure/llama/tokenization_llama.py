# Hugging Face 的 tokenizers 库（底层的 Rust 实现库，速度很快）

# Tokenizer: 主类，用于加载和使用分词器
# decoders: 解码器，用于将 token ID 转换回文本
# pre_tokenizers: 在 BPE 算法运行前，先把句子切碎（比如按空格切分）
from tokenizers import Tokenizer, decoders, pre_tokenizers

# 导入 BPE 模型算法类，这是 Llama 使用的核心分词算法
from tokenizers.models import BPE

# 导入 Transformers 库内部的辅助工具
# _get_prepend_scheme: 用于处理 "是否在句子开头加空格" 的逻辑
# TokenizersBackend: 这是一个基类，Hugging Face 用它来桥接 Python 顶层接口和 Rust 底层实现
from transformers.tokenization_utils_base import _get_prepend_scheme
from transformers.tokenization_utils_tokenizers import TokenizersBackend
from transformers.utils import logging


logger = logging.get_logger(__name__)

# 定义文件名映射。
# 当用户调用 tokenizer.save_pretrained() 时：
# 1. 词表内容会被保存为 tokenizer.model (这是 SentencePiece 格式的二进制或文本模型)
# 2. 完整的分词器配置（包括前后处理逻辑）会被保存为 tokenizer.json

# 当需要加载词表时，默认去找名为 tokenizer.model 的文件；
# 当需要加载完整的 tokenizer 配置时，去找 tokenizer.json。
# 这在 from_pretrained 加载模型时非常关键。
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}

# tokenizer_config.json 也很重要
# 主要确定的是一些基础的配置

# Begin Instruction，End Instruction
B_INST, E_INST = "[INST]", "[/INST]"
# Begin System，End System
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# # fmt: off / # fmt: on: 这不是 Python 代码，而是给代码格式化工具（如 Black, autopep8）看的指令。
# 通常是为了保持长字符串的排版不被破坏。
# fmt: off
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
# fmt: on


# 继承自 TokenizersBackend，使用 Rust 实现的 tokenizers 库
# tokenizer 算法使用 rust 实现，比纯用 python 快很多
class LlamaTokenizer(TokenizersBackend):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

    This uses notably ByteFallback and no normalization.

    使用了 ByteFallback（处理生僻字转 unicode 字节）
    且没有做 Normalization（比如不自动转小写）

    ```python
    >>> from transformers import LlamaTokenizer

    >>> tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    >>> tokenizer.encode("Hello this is a test")
    [1, 15043, 445, 338, 263, 1243]
    ```

    If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or
    call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the
    values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
    [post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) documentation.


    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab (`str`, `dict` or `list`, *optional*):
            Path to the vocabulary file, a dictionary or a list of tokens.
        merges (`str` or `list`, *optional*):
            Path to the merges file or a list of merges.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            Whether or not the default system prompt for Llama should be used
        add_prefix_space (`bool`, *optional*):
            Whether or not the tokenizer should automatically add a prefix space
    """

    # 将前面定义的常量绑定到类属性，方便父类方法调用查找文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 默认 padding 方向为左边
    # 这样生成的 token 永远在最右边，位置对齐更方便）
    # 不过也有一些是设置在 right，比如 llama chat 系列
    padding_side = "left"
    # 告诉模型，调用 tokenizer 后返回的字典里应该包含哪些字段
    # input_ids: token 的数字 ID 列表
    # attention_mask: 告诉模型哪些是真实 token (1)，哪些是 padding (0)
    model_input_names = ["input_ids", "attention_mask"]

    # 指定底层算法模型为 BPE
    model = BPE

    def __init__(
        self,
        # vocab 和 merges 是 BPE 算法需要的词表和合并规则
        # 这个一般会在 tokenizer.json 中会定义
        vocab: str | dict | list | None = None,
        merges: str | list | None = None,
        # 是否在解码后清理多余空格
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        use_default_system_prompt=False,
        legacy=False, # 兼容旧版本的标志位
        add_prefix_space=None,  # 是否在句子开头自动加一个空格（Llama 的 BPE 经常需要这个）
        **kwargs,
    ):
        # 如果用户没传 add_prefix_space，默认为 True
        self.add_prefix_space = add_prefix_space if add_prefix_space is not None else True
        self.legacy = legacy
        self._vocab = vocab

        # # 如果没有提供词表，手动创建一个极简词表防止报错
        if vocab is None:
            self._vocab = {
                str(unk_token): 0,
                str(bos_token): 1,
                str(eos_token): 2,
            }

        self._merges = merges or []
        # 实例化底层的 Rust Tokenizer 对象
        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab, merges=self._merges, 
                fuse_unk=True, # 融合未知词逻辑
                byte_fallback=True, # 如果遇到词表中没有的字符，分解为字节处理（支持生僻字）
                dropout=None
            )
        )

        # Llama 不做文本规范化（如 unicode normalization, lower casing），保持原样
        self._tokenizer.normalizer = None

        # 设置预分词器 (Pre-tokenizer)
        # Metaspace 是一种常见的预分词策略（类似 SentencePiece 的处理方式）
        # 它将空格替换为特殊的下划线字符（这里用的是 ▁）
        # prepend_scheme 决定是否在句子的最前面加个 "▁"
        # 输入："Hello world"
        # 变换后："▁Hello▁world"
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme=_get_prepend_scheme(self.add_prefix_space, self), split=False
        )

        # 设置解码器 (Decoder) 序列：
        # 1. 把特殊的 "▁" 替换回空格
        # 2. ByteFallback: 把字节 ID 还原回字符
        # 3. Fuse: 把碎片拼起来
        sequence = [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]

        # 如果之前为了处理加了前缀空格，解码的时候要把这个多余的空格去掉
        if self.add_prefix_space:
            # 只删除左边（开头）的 1 个
            sequence += [decoders.Strip(content=" ", left=1)]

        self._tokenizer.decoder = decoders.Sequence(sequence)


        self.use_default_system_prompt = use_default_system_prompt
        super().__init__(
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            use_default_system_prompt=use_default_system_prompt,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )


__all__ = ["LlamaTokenizer", "LlamaTokenizerFast"]

# Backward alias
# 为了兼容性，把 LlamaTokenizerFast 指向 LlamaTokenizer
# 在新版 Transformers 中，通常会有 Tokenizer(纯Python) 和 TokenizerFast(Rust) 两个类
# 这里直接把 Fast 的名字也给了它，说明这个类本身就是基于 Rust 后端的
LlamaTokenizerFast = LlamaTokenizer
