import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_k(S: torch.Tensor, threshold: float) -> int:
    total = S.sum()
    cumulative = torch.cumsum(S, dim=0)
    return torch.sum(cumulative < threshold * total).item()

class SARA(nn.Module):
    def __init__(self, pretrained_weight: torch.Tensor, threshold: float = 0.01):
        super(SARA, self).__init__()
        self.original_weight = nn.Parameter(pretrained_weight, requires_grad=False)
        self.threshold = threshold
        
        U, S, V = torch.svd(pretrained_weight)
        self.k = calculate_k(S, threshold)
        
        self.U_k = nn.Parameter(torch.randn(pretrained_weight.shape[0], self.k))
        self.V_k = nn.Parameter(torch.randn(self.k, pretrained_weight.shape[1]))
        self.lambda_k = nn.Parameter(torch.randn(self.k))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_W = self.U_k @ torch.diag(self.lambda_k) @ self.V_k
        return x @ (self.original_weight + delta_W).t()

class MoSARA(nn.Module):
    def __init__(self, pretrained_weight: torch.Tensor, threshold: float = 0.5, num_experts: int = 5):
        super(MoSARA, self).__init__()
        self.original_weight = nn.Parameter(pretrained_weight, requires_grad=False)
        self.threshold = threshold
        self.num_experts = num_experts
        
        U, S, V = torch.svd(pretrained_weight)
        self.k = calculate_k(S, threshold)
        
        self.U_k = nn.Parameter(U[:, :self.k], requires_grad=False)
        self.V_k = nn.Parameter(V[:, :self.k], requires_grad=False)
        self.lambda_k = nn.Parameter(torch.randn(num_experts, self.k))
        self.v = nn.Parameter(torch.zeros(pretrained_weight.shape[1]))
        
        self.W_g1 = nn.Parameter(torch.randn(self.k, 1))
        self.W_g2 = nn.Parameter(torch.randn(1, num_experts))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing_input = x @ self.U_k
        g = F.softmax(routing_input @ self.W_g1 @ self.W_g2, dim=-1)
        
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            delta_W = self.U_k @ torch.diag(self.lambda_k[i]) @ self.V_k.t()
            expert_output = x @ (self.original_weight + delta_W).t()
            output += g[:, i].unsqueeze(1) * expert_output
        
        output = output * (1 + self.v)
        return output

class SARALlamaModel(nn.Module):
    def __init__(self, base_model: LlamaForCausalLM, use_mo_sara: bool = False, threshold: float = 0.01, num_experts: int = 5):
        super(SARALlamaModel, self).__init__()
        self.base_model = base_model
        self.use_mo_sara = use_mo_sara
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                if 'q_proj' in name or 'v_proj' in name:
                    if use_mo_sara:
                        setattr(module, 'weight', MoSARA(module.weight, threshold, num_experts))
                    else:
                        setattr(module, 'weight', SARA(module.weight, threshold))
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.base_model(input_ids, attention_mask=attention_mask)

def load_and_adapt_model(model_name: str, use_mo_sara: bool = False, threshold: float = 0.01, num_experts: int = 5, device: str = 'cuda') -> Tuple[SARALlamaModel, LlamaTokenizer]:
    try:
        base_model = LlamaForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        adapted_model = SARALlamaModel(base_model, use_mo_sara, threshold, num_experts).to(device)
        return adapted_model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def test_model(model: SARALlamaModel, tokenizer: LlamaTokenizer, input_text: str, device: str = 'cuda') -> str:
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    output = model.generate(input_ids, max_length=50)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    model_name = "decapoda-research/llama-7b-hf"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("Testing SARA:")
    sara_model, tokenizer = load_and_adapt_model(model_name, use_mo_sara=False, device=device)
    sara_output = test_model(sara_model, tokenizer, "Once upon a time", device=device)
    logger.info(f"SARA output: {sara_output}")
    
    logger.info("\nTesting Mo-SARA:")
    mo_sara_model, _ = load_and_adapt_model(model_name, use_mo_sara=True, device=device)
    mo_sara_output = test_model(mo_sara_model, tokenizer, "Once upon a time", device=device)
    logger.info(f"Mo-SARA output: {mo_sara_output}")
    
    sara_params = sum(p.numel() for p in sara_model.parameters() if p.requires_grad)
    mo_sara_params = sum(p.numel() for p in mo_sara_model.parameters() if p.requires_grad)
    logger.info(f"\nNumber of trainable parameters in SARA: {sara_params}")
    logger.info(f"Number of trainable parameters in Mo-SARA: {mo_sara_params}")

if __name__ == "__main__":
    main()