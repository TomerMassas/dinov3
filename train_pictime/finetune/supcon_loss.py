import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Expects L2-normalized features and integer labels.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] L2-normalized embeddings.
            labels:   [B] integer identity IDs.
        Returns:
            Scalar loss.
        """
        device = features.device
        B = features.shape[0]

        # Pairwise cosine similarity (features are already L2-normed)
        sim = features @ features.T / self.temperature  # [B, B]

        # Positive mask: same label, exclude self
        labels = labels.unsqueeze(0)  # [1, B]
        pos_mask = (labels == labels.T).float()  # [B, B]
        pos_mask.fill_diagonal_(0.0)

        # For numerical stability, subtract max per row
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Exclude self from denominator
        self_mask = torch.eye(B, device=device)
        denom = torch.exp(sim) * (1.0 - self_mask)
        log_denom = torch.log(denom.sum(dim=1, keepdim=True) + 1e-8)

        # Per-positive log-softmax
        log_prob = sim - log_denom  # [B, B]

        # Mean over positives for each anchor
        num_positives = pos_mask.sum(dim=1)  # [B]
        valid = num_positives > 0
        loss = -(pos_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)
        loss = loss[valid].mean()

        return loss
