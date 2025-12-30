from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.qwen3.configuration_qwen3 import Qwen3Config
from fla.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3ForCausalLM

AutoConfig.register(Qwen3Config.model_type, Qwen3Config, exist_ok=True)
AutoModel.register(Qwen3Config, Qwen3Model, exist_ok=True)
AutoModelForCausalLM.register(Qwen3Config, Qwen3ForCausalLM, exist_ok=True)

__all__ = ['Qwen3Config', 'Qwen3ForCausalLM', 'Qwen3Model']