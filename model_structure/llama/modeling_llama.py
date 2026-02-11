from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

from transformers.activations import ACT2FN # 激活函数映射表
from transformers.cache_utils import Cache, DynamicCache # KV Cache 相关
from transformers.generation import GenerationMixin # 生成能力 mixin
from transformers.integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func # 内核集成
from transformers.masking_utils import create_causal_mask # 因果 mask 创建
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
) # 通用问答、序列分类、token 分类和梯度检查点层
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
) # 基本模型输出和因果语言模型输出
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update # RoPE 初始化函数
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel # 所有注意力函数和预训练模型基类
from transformers.processing_utils import Unpack # 解包工具
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging # 工具类
from transformers.utils.generic import check_model_inputs, maybe_autocast # 通用工具
try:
    from .configuration_llama import LlamaConfig
except ImportError:
    from configuration_llama import LlamaConfig

logger = logging.get_logger(__name__) # 初始化日志

# 如果环境中安装了高度优化的 kernels，则在运行 forward 时优先使用从 Hub 加载的高性能 CUDA Kernel
# 而不是下面 Python 写的原生 PyTorch 代码。
@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # 可学习参数，所有维度的缩放因子初始都设为 1
        self.variance_epsilon = eps # 一个很小的数，防止除以 0

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32) # 为了计算准确，必须先转为 float32
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        # 当你使用 print(model) 查看架构时，直接看到该层的维度和 eps 的具体值
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # Llama 2 通常是 default
        # 而 Llama 3 或长文本模型可能会使用 llama3、linear 或 dynamic
        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            # 从 ROPE_INIT_FUNCTIONS 字典中寻找对应的算法。
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        # persistent=False 表示这个参数虽然随模型移动（比如从 CPU 到 GPU）
        # 但不会被保存在模型的 state_dict（权重文件）中
        # 通过 register_buffer 和缓存机制，避免了在每次推理时重复计算复杂的幂运算。
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    # 计算每个维度旋转的“速度”
    @staticmethod
    def compute_default_rope_parameters(
        config: LlamaConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        # 旋转基数，通常是 10000（Llama 3 为 500000）。
        # 基数越大，高维度的旋转速度越慢，适合处理更长的上下文。
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        # 默认的是不带缩放的 RoPE，Llama 3 或者是其他支持更长的上下文处理的模型需要使用缩放因子。
        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        # 计算逆频率
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad() # 计算位置编码时不需要记录梯度，不参与参数更新
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # inv_freq: [dim/2] 这里的 dim 是 head_dim
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        # inv_freq_expanded: [batch_size, dim/2, 1]
        # position_ids: [batch_size, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        # position_ids_expanded: [batch_size, 1, seq_len]

        # mps 代表 Apple Silicon 系列的芯片，将它当作 CPU 处理
        # 因为 mps 可能还不支持 autocast 自动混合精度计算
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        # 强制使用 float32 精度计算
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            # [batch_size, dim/2, seq_len] 到 [batch_size, seq_len, dim/2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1) # [batch_size, seq_len, dim]
            cos = emb.cos() * self.attention_scaling # 可能会乘以缩放因子
            sin = emb.sin() * self.attention_scaling # 可能会乘以缩放因子

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype) # 转换回原来的精度


# 这是一个非常巧妙的配合使用 RoPE 算法，如果一正一负内存中不连续，计算效率略低。
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # [d_1, d_2, d_3, d_4, d_5, d_6]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # -> [-d_4, -d_5, -d_6, d_1, d_2, d_3]
    return torch.cat((-x2, x1), dim=-1)

# 如果可以，优先使用 CUDA 内核中的 rotary_pos_emb 函数，否则使用 torch 实现
@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim 参数指定了对 cos[position_ids] 和 sin[position_ids] 进行维度扩充的轴
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # cos 和 sin 的原始形状通常是 [batch_size, seq_len, head_dim]
    # q，k 的形状往往多了一个“多头”维度，[batch_size, num_heads, seq_len, head_dim]
    # unsqueeze_dim=1，就是把 cos 和 sin 的维度从 2 扩展到 3，变成 [batch_size, 1, seq_len, head_dim]
    # 这样就可以和 q，k 进行广播相乘
    # [batch_size, seq_len, head_dim] -> [batch_size, 1, seq_len, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# 采用 SwiGLU 激活函数的 MLP，具有一个额外的门控机制
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# 在 GQA 或者是 MQA 中，query 的个数通常比 key 和 value 多
# 因此需要将 key 和 value 的维度重复多遍
# [batch_size, num_key_value_heads, seq_len, head_dim] 
# -> [batch_size, num_attention_heads, seq_len, head_dim]
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# eager：急切版本，没有调用底层高度优化的硬件加速内核
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    """
    该函数实现了注意力机制的核心逻辑，包括 KV 头的重复、注意力权重的计算、掩码应用以及 Softmax 归一化。

    Args:
        module (`nn.Module`): 
            调用此函数的注意力模块实例。需包含属性 `num_key_value_groups` (Q 头数 / KV 头数的比例)
            以及 `training` (布尔值，决定是否应用 dropout)。
        query (`torch.Tensor`): 
            [batch_size, num_heads, query_sequence_length, head_dim]
        key (`torch.Tensor`): 
            [batch_size, num_key_value_heads, key_sequence_length, head_dim]
        value (`torch.Tensor`): 
            [batch_size, num_key_value_heads, key_sequence_length, head_dim]
        attention_mask (`torch.Tensor`, *optional*): 
            [batch_size, num_heads, query_sequence_length, -1]
            一般是一个很大的下三角矩阵
            num_heads 通常是 1，到时候直接广播就行
        scaling (`float`): 
            缩放系数，通常为 `1 / sqrt(head_dim)`。
        dropout (`float`, *optional*, 默认为 0.0): 
            在 Softmax 之后应用的 Dropout 概率。
        **kwargs: 
            其他传递给 Transformer 层的关键字参数。

    注意：
        训练阶段或者 Prefill 阶段，query_sequence_length 和 key_sequence_length 通常是相等的
        （KV Cache）
        在 Decode 阶段，query_sequence_length 通常是 1，而 key_sequence_length 是整个序列的长度
    """
    # [batch_size, num_key_value_heads, key_sequence_length, head_dim] 
    # -> [batch_size, num_heads, key_sequence_length, head_dim]
    key_states = repeat_kv(key, module.num_key_value_groups)
    # [batch_size, num_key_value_heads, key_sequence_length, head_dim] 
    # -> [batch_size, num_heads, key_sequence_length, head_dim]
    value_states = repeat_kv(value, module.num_key_value_groups)

    # [batch_size, num_heads, query_sequence_length, head_dim] @ [batch_size, num_heads, head_dim, key_sequence_length]
    # -> [batch_size, num_heads, query_sequence_length, key_sequence_length]
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    # Causal Mask 因果掩码，Train 的时候需要，Decode 的时候一般不需要
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # softmax 设计 e 的指数运算，使用 float32 避免数值溢出
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # training 为 true 的时候需要 dropout 防止过拟合，decode 的时候不需要
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    # [batch_size, num_heads, query_sequence_length, key_sequence_length] @ [batch_size, num_heads, key_sequence_length, head_dim]
    # -> [batch_size, num_heads, query_sequence_length, head_dim]
    attn_output = torch.matmul(attn_weights, value_states)
    # [batch_size, num_heads, query_sequence_length, head_dim] 
    # -> [batch_size, query_sequence_length, num_heads, head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()
    # 方便后续 num_heads 和 head_dim 合并
    # 注意：transpose 后一般都需要 contiguous() 来保证内存连续，否则后面的 view 或 reshape 合并会报错

    return attn_output, attn_weights


# use_kernelized_func 用于装饰类
# use_kernel_func_from_hub 用于装饰函数
@use_kernelized_func(apply_rotary_pos_emb)
class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        # 当前层的索引
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        # 标记为因果掩码
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor, # [batch_size, seq_len, hidden_size]
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, # (cos, sin) [batch_size, seq_len, head_dim]
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None, # 当前 token 在序列中的绝对位置索引
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 一般在 Decode 阶段的时候，seq_len=1，其他时候 seq_len 才是整个序列的长度
        input_shape = hidden_states.shape[:-1] # [batch_size, seq_len]
        hidden_shape = (*input_shape, -1, self.head_dim) # [batch_size, seq_len, -1, head_dim]

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) # [batch_size, num_attention_heads, seq_len, head_dim]
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2) # [batch_size, num_key_value_heads, seq_len, head_dim]
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # [batch_size, num_key_value_heads, seq_len, head_dim]

        cos, sin = position_embeddings # [batch_size, seq_len, head_dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None: # 如果已经开始 decode 阶段，KV 缓存不为空
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # 某些特殊的 Cache 实现可能需要在缓存内部对数据进行反旋转或重新旋转处理，或者用于相对位置计算。
            # 为什么传 cache_position？ 
            # 对于 StaticCache（静态显存分配），模型需要知道当前 Token 应该填入缓存矩阵的哪一行
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # 拿出所有的 KV 缓存
            # 此时的 seq_len 才会变成整个序列的长度
            # self.layer_idx：告诉缓存对象，这是第几层的缓存
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # query_states: [batch_size, num_attention_heads, query_seq_len, head_dim]
        # key_states: [batch_size, num_key_value_heads, key_seq_len, head_dim]
        # value_states: [batch_size, num_key_value_heads, key_seq_len, head_dim]
        # attention_mask: 很大的一个下三角矩阵

        # 动态选择数学算子来计算 Attention
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs, # 一些 attention 的计算方法可能需要一些额外的参数，比如 use_cache 等
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output) # [batch_size, query_seq_len, hidden_size]
        return attn_output, attn_weights # attn_weights: [batch_size, num_heads, query_seq_len, key_seq_len]


# 这是一个用于 节省显存 的机制，在训练时通过“重计算”中间变量来减少内存占用。
# 这意味着该模块支持 梯度检查点（Gradient Checkpointing），适合训练超大规模模型。
# 在 HuggingFace Transformers 中，这是常见做法，尤其对 LLM 训练至关重要。
class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        # 输入归一化层。用于在进入 Attention 之前对数据进行标准化。
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 注意力后归一化层。用于在进入 MLP 之前对数据进行标准化。
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs], # Hugging Face 定义了一个 TypedDict，名叫 TransformersKwargs
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# @auto_docstring: 装饰器，用于自动生成文档字符串。它会从父类或配置文件中提取信息
@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    # 定义了在 state_dict（权重字典）中，核心模型部分的命名前缀。
    # 例如，LlamaForCausalLM 的权重字典里
    # 主干部分的参数名通常是 model.layers.0...，而不是直接 layers.0...。
    # 这个 "model" 就是在这里定义的
    supports_gradient_checkpointing = True
    # 开启后，模型在训练时会不保存中间激活值，而是在反向传播时重新计算它们
    _no_split_modules = ["LlamaDecoderLayer"] # 一个 LlamaDecoderLayer 必须完整地放在同一张 GPU 上，不可以进行切分
    _skip_keys_device_placement = ["past_key_values"]
    # 告诉 accelerate 库，不要尝试自动移动 past_key_values（KV Cache）到特定设备。
    _supports_flash_attn = True
    # 告诉 accelerate 库，支持 FlashAttention 优化
    _supports_sdpa = True
    # 告诉 accelerate 库，支持缩放点积注意力（Scaled Dot-Product Attention）优化
    _supports_flex_attn = True
    # 告诉 accelerate 库，支持灵活注意力（Flexible Attention）优化

    _can_compile_fullgraph = True
    # 告诉 accelerate 库，支持编译整个模型的计算图
    _supports_attention_backend = True
    # 告诉 accelerate 库，支持注意力机制的后端
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }
    # 定义了当用户要求输出中间层结果（如 output_hidden_states=True 或 output_attentions=True）时，应该从哪些模块抓取数据。
    # "hidden_states" 对应 LlamaDecoderLayer：每经过一层 Decoder，就抓取一次输出。
    # "attentions" 对应 LlamaAttention：在 Attention 层内部计算出权重矩阵后，抓取它。


@auto_docstring
class LlamaModel(LlamaPreTrainedModel): # LlamaPreTrainedModel 是 Llama 的父类，主要用于管理 Model 权重和梯度的保存和加载
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 设置 self.padding_idx
        # 初始化时，padding token 对应的嵌入向量全为 0
        # 训练过程中，这个位置的参数不会被更新，避免填充值影响模型训练
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False # 保存所有中间激活值，反向传播直接用，速度快但显存占用高

        # Initialize weights and apply final processing
        self.post_init() # 递归地初始化所有子模块的权重

    @check_model_inputs # 检查输入参数是否合法
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast: # 这是 Hugging Face 定义的一个标准输出类，包含 last_hidden_state 和 past_key_values
        # 必须且只能传一个
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            # 如果启用了 Cache，但没有传入 past_key_values，则创建一个新的 DynamicCache
            # Hugging Face 新版的高效缓存实现，可以动态增长
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            # 当前输入 Token 在完整序列中的绝对位置
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            # 首个 Token（Prefill 阶段）： [0, 1, ..., seq_len-1]
            # 生成 Token（Decode 阶段）： past_seen_tokens 比如是 10，当前输入长度是 1，那位置就是 [10]
            cache_position: torch.Tensor = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) # [1, seq_len]

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        # [B, 1, Q, K] 下三角矩阵，Prefill 阶段
        # [B, 1, 1, K] 扁长的条形，Decode 阶段

        hidden_states = inputs_embeds # [batch_size, seq_len, hidden_size]
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids) # [batch_size, seq_len, hidden_size]

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# 在基础的 LlamaModel 之上加了一个“LM Head”
# GenerationMixin 提供了生成文本的功能，给模型赋予了 .generate() 方法
@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    # lm_head.weight 和 embed_tokens.weight 共享参数
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    # 按列切分，最后收集输出
    _tp_plan = {"lm_head": "colwise_gather_output"}
    # 流水线并行：lm_head 接收 hidden_states，输出 logits
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple # 为了适配 return_dict=False 的情况下，返回元组而不是 Dict
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 如果传入了 labels，说明现在是训练阶段
        loss = None
        if labels is not None:
            # self.loss_function 是父类提供的一个通用方法，支持标签平滑、类别加权等功能
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        # 使用 CausalLMOutputWithPast 封装输出
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 其他几个任务需要实现这几个类
# 在官方 transformers 库中，这些类也是留空的
class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]


if __name__ == "__main__":
    # 查看 LlamaModel 的文档字符串
    print(LlamaModel.__doc__)
    

