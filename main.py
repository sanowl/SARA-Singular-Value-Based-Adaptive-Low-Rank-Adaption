import torch
import torch.nn as nn
import torch.nn.functional as F

class SARA(nn.Module):
    def __init__(self, pretrained_weight, threshold=0.01):
        super(SARA, self).__init__()
        self.original_weight = nn.Parameter(pretrained_weight, requires_grad=False)
        self.threshold = threshold
        
        U, S, V = torch.svd(pretrained_weight)
        total = S.sum()
        cumulative = torch.cumsum(S, dim=0)
        self.k = torch.sum(cumulative < threshold * total).item()
        
        self.U_k = nn.Parameter(torch.randn(pretrained_weight.shape[0], self.k))
        self.V_k = nn.Parameter(torch.randn(self.k, pretrained_weight.shape[1]))
        self.lambda_k = nn.Parameter(torch.randn(self.k))
        
    def forward(self, x):
        delta_W = self.U_k @ torch.diag(self.lambda_k) @ self.V_k
        return F.linear(x, self.original_weight + delta_W)

class MoSARA(nn.Module):
    def __init__(self, pretrained_weight, threshold=0.5, num_experts=5):
        super(MoSARA, self).__init__()
        self.original_weight = nn.Parameter(pretrained_weight, requires_grad=False)
        self.threshold = threshold
        self.num_experts = num_experts
        
        U, S, V = torch.svd(pretrained_weight)
        total = S.sum()
        cumulative = torch.cumsum(S, dim=0)
        self.k = torch.sum(cumulative < threshold * total).item()
        
        self.U_k = nn.Parameter(U[:, :self.k], requires_grad=False)
        self.V_k = nn.Parameter(V[:self.k, :], requires_grad=False)
        self.lambda_k = nn.Parameter(torch.randn(num_experts, self.k))
        self.v = nn.Parameter(torch.zeros(pretrained_weight.shape[0]))
        
        self.router_W1 = nn.Parameter(torch.randn(self.k, 1))
        self.router_W2 = nn.Parameter(torch.randn(1, num_experts))
        
    def forward(self, x):
        # Compute routing weights
        routing_input = x @ self.U_k
        g = F.softmax(routing_input @ self.router_W1 @ self.router_W2, dim=-1)
        
        # Compute weighted sum of expert outputs
        delta_W = self.U_k @ torch.diag_embed(self.lambda_k) @ self.V_k
        expert_outputs = F.linear(x.unsqueeze(1), self.original_weight + delta_W)
        output = torch.sum(expert_outputs * g.unsqueeze(-1), dim=1)
        
        # Apply diagonal scaling
        output = output * (1 + self.v)
        
        return output

# Usage example
pretrained_weight = torch.randn(768, 768)  # Example pretrained weight matrix

# SARA
sara_layer = SARA(pretrained_weight, threshold=0.01)
x = torch.randn(32, 768)  # Example input
output_sara = sara_layer(x)

# Mo-SARA
mo_sara_layer = MoSARA(pretrained_weight, threshold=0.5, num_experts=5)
output_mo_sara = mo_sara_layer(x)