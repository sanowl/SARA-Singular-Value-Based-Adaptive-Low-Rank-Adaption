import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer

def calculate_k(S, threshold):
    total = S.sum()
    cumulative = torch.cumsum(S, dim=0)
    return torch.sum(cumulative < threshold * total).item()

class SARA(nn.Module):
    def __init__(self, pretrained_weight, threshold=0.01):
        super(SARA, self).__init__()
        self.original_weight = nn.Parameter(pretrained_weight, requires_grad=False)
        self.threshold = threshold
        
        U, S, V = torch.svd(pretrained_weight)
        self.k = calculate_k(S, threshold)
        
        self.U_k = nn.Parameter(torch.randn(pretrained_weight.shape[0], self.k))
        self.V_k = nn.Parameter(torch.randn(self.k, pretrained_weight.shape[1]))
        self.lambda_k = nn.Parameter(torch.randn(self.k))
        
    def forward(self, x):
        delta_W = self.U_k @ torch.diag(self.lambda_k) @ self.V_k
        return x @ (self.original_weight + delta_W).t()

class MoSARA(nn.Module):
    def __init__(self, pretrained_weight, threshold=0.5, num_experts=5):
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
        
    def forward(self, x):
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
    def __init__(self, base_model, use_mo_sara=False, threshold=0.01, num_experts=5):
        super(SARALlamaModel, self).__init__()
        self.base_model = base_model
        self.use_mo_sara = use_mo_sara
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                if 'q_proj' in name or 'v_proj' in name:
                    if use_mo_sara:
                        setattr(module, 'weight', MoSARA(module.weight, threshold, num_experts))
                    else:
                        setattr(modulje, 'weight', SARA(module.weight, threshold))
    
    def forward(self, input_ids, attention_mask=None):
        return self.base_model(input_ids, attention_mask=attention_mask)

def load_and_adapt_model(model_name, use_mo_sara=False, threshold=0.01, num_experts=5):
    base_model = LlamaForCausalLM.from_pretrained(model_name)
    adapted_model = SARALlamaModel(base_model, use_mo_sara, threshold, num_experts)
    return adapted_model

def test_model(model, tokenizer, input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=50)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    model_name = "decapoda-research/llama-7b-hf" 
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    
    print("Testing SARA:")
    sara_model = load_and_adapt_model(model_name, use_mo_sara=False)
    sara_output = test_model(sara_model, tokenizer, "Once upon a time")
    print(f"SARA output: {sara_output}")
    
    print("\nTesting Mo-SARA:")
    mo_sara_model = load_and_adapt_model(model_name, use_mo_sara=True)
    mo_sara_output = test_model(mo_sara_model, tokenizer, "Once upon a time")
    print(f"Mo-SARA output: {mo_sara_output}")
    
    # Print number of trainable parameters
    sara_params = sum(p.numel() for p in sara_model.parameters() if p.requires_grad)
    mo_sara_params = sum(p.numel() for p in mo_sara_model.parameters() if p.requires_grad)
    print(f"\nNumber of trainable parameters in SARA: {sara_params}")
    print(("test    "))
    print(f"Number of trainable parameters in Mo-SARA: {mo_sara_params}")