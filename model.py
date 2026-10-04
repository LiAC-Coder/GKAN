import math
import torch
from torch import nn
from .kan_experts import KANExpertNetwork


def kl_to_uniform(probabilities):

    log_p = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    return (probabilities * (log_p + math.log(probabilities.shape[-1]))).sum(-1)


class NeighborhoodMoE(nn.Module):


    def __init__(self, dim, num_experts=3, **ken_options):
        super().__init__()
        self.experts = nn.ModuleList(
            [KANExpertNetwork(dim, **ken_options) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, aggregated):
        weights = self.gate(aggregated).softmax(-1)
        result = torch.zeros_like(aggregated)
        for index, expert in enumerate(self.experts):
            result = result + weights[:, index:index + 1] * expert(aggregated)
        return result, weights


class GraphAwareMoELayer(nn.Module):
    def __init__(self, dim, num_experts=3, **ken_options):
        super().__init__()
        self.short_moe = NeighborhoodMoE(dim, num_experts, **ken_options)
        self.long_moe = NeighborhoodMoE(dim, num_experts, **ken_options)
        self.high_experts = nn.ModuleList(
            [KANExpertNetwork(dim, **ken_options) for _ in range(2)])
        self.high_gate = nn.Linear(dim, 2, bias=False)

    def forward(self, embedding, short_adj, long_adj):
        short_low, short_weights = self.short_moe(torch.sparse.mm(short_adj, embedding))
        long_low, long_weights = self.long_moe(torch.sparse.mm(long_adj, embedding))
        high_weights = self.high_gate(short_low + long_low).softmax(-1)
        result = (high_weights[:, :1] * self.high_experts[0](short_low)
                  + high_weights[:, 1:] * self.high_experts[1](long_low))
        low_loss = kl_to_uniform(short_weights) + kl_to_uniform(long_weights)
        high_loss = kl_to_uniform(high_weights)
        return result, low_loss, high_loss


class GamerKAN(nn.Module):
    def __init__(self, args, num_user, num_item, short_adj, long_adj):
        super().__init__()
        if getattr(args, "pre_model", 0) != 0:
            raise ValueError("Invalid input")
        self.num_user, self.num_item = num_user, num_item
        self.user_emb = nn.Parameter(torch.empty(num_user, args.dim))
        self.item_emb = nn.Parameter(torch.empty(num_item, args.dim))
        nn.init.xavier_uniform_(self.user_emb)
        nn.init.xavier_uniform_(self.item_emb)

        self.register_buffer("short_adj", short_adj, persistent=False)
        self.register_buffer("long_adj", long_adj, persistent=False)
        self.layers = nn.ModuleList([
            GraphAwareMoELayer(args.dim, args.num_experts, resolution=args.resolution,
                              spline_order=args.spline_order,
                              activation_checkpoint=args.activation_checkpoint)
            for _ in range(args.layers)
        ])

    def forward(self, return_ladapter=True):
        embedding = torch.cat((self.user_emb, self.item_emb))
        pooled = torch.zeros_like(embedding)
        low_loss = embedding.new_zeros(len(embedding))
        high_loss = embedding.new_zeros(len(embedding))
        for layer in self.layers:
            embedding, low, high = layer(embedding, self.short_adj, self.long_adj)

            pooled = pooled + embedding
            low_loss, high_loss = low_loss + low, high_loss + high
        user, item = pooled[:self.num_user], pooled[self.num_user:]
        if return_ladapter:
            return user, item, low_loss, high_loss
        return user, item

    @staticmethod
    def get_logits(user_embedding, item_embedding, users, items):
        device = user_embedding.device
        users = torch.as_tensor(users, dtype=torch.long, device=device)
        items = torch.as_tensor(items, dtype=torch.long, device=device)
        return (user_embedding[users] * item_embedding[items]).sum(-1)

    def get_rating(self, user_embedding, item_embedding, users, items):
        return self.get_logits(user_embedding, item_embedding, users, items).sigmoid()
