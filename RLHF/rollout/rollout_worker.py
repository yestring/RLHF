from vllm import LLM, SamplingParams


class RolloutWorker:
    """
    推理工作线程
    
    使用 vLLM 进行高效推理：
    - Prefix caching for performance
    - Batch processing
    - Configurable sampling
    """
    
    def __init__(self, model_name, enable_prefix_caching=True):
        """
        初始化 RolloutWorker
        
        Args:
            model_name: 模型名称
            enable_prefix_caching: 是否启用前缀缓存
        """
        self.engine = LLM(
            model=model_name,
            enable_prefix_caching=enable_prefix_caching,
            trust_remote_code=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=128
        )

    def generate(self, prompts):
        """
        批量生成文本
        
        Args:
            prompts: 提示词列表
            
        Returns:
            生成的文本列表
        """
        outputs = self.engine.generate(
            prompts,
            self.sampling_params
        )
        
        responses = []
        for output in outputs:
            responses.append(
                output.outputs[0].text
            )
        
        return responses
    
    def generate_with_params(self, prompts, temperature=0.8, top_p=0.95, max_tokens=128):
        """
        使用自定义参数生成文本
        
        Args:
            prompts: 提示词列表
            temperature: 采样温度
            top_p: top-p 采样参数
            max_tokens: 最大生成令牌数
            
        Returns:
            生成的文本列表
        """
        custom_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        outputs = self.engine.generate(prompts, custom_params)
        
        responses = []
        for output in outputs:
            responses.append(output.outputs[0].text)
        
        return responses
