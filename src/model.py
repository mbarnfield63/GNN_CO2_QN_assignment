import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from tqdm import tqdm


class CO2AssignmentGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        num_isotopes,
        num_classes,
        hidden_dim=256,
        embed_dim=8,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate

        # Structural embedding for isotope context
        self.iso_embed = nn.Embedding(num_isotopes, embed_dim)
        total_input_dim = input_dim + embed_dim

        # Project input immediately to hidden_dim to enable residual additions
        self.input_proj = nn.Linear(total_input_dim, hidden_dim)

        # 2 Layers of GraphSAGE for long-range polyad resolution
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # LayerNorms stabilize training in deep GNNs
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, edge_index, iso_idx):
        iso_emb = self.iso_embed(iso_idx)
        x = torch.cat([x, iso_emb], dim=1)

        # Initial projection
        x = self.input_proj(x)

        # Layer 1 with Residual Connection
        h = self.conv1(x, edge_index)
        h = self.ln1(h)
        h = F.gelu(h)
        x = x + F.dropout(h, p=self.dropout_rate, training=self.training)

        # Layer 2 with Residual Connection
        h = self.conv2(x, edge_index)
        h = self.ln2(h)
        h = F.gelu(h)
        x = x + F.dropout(h, p=self.dropout_rate, training=self.training)

        return self.head(x)

    def mc_dropout_predict(self, data, device, num_samples=30):
        """Runs stochastic forward passes on the full graph."""
        self.train()  # Force dropout ON
        num_classes = self.head[-1].out_features
        num_nodes = data.x.shape[0]
        data = data.to(device)

        mean_probs = torch.zeros((num_nodes, num_classes), dtype=torch.float32, device="cpu")
        sq_probs = torch.zeros((num_nodes, num_classes), dtype=torch.float32, device="cpu")
        mean_entropy = torch.zeros(num_nodes, dtype=torch.float32, device="cpu")

        with torch.no_grad():
            for _ in tqdm(range(num_samples), desc="MC Dropout Inference"):
                logits = self.forward(data.x, data.edge_index, data.iso_idx)
                probs = F.softmax(logits, dim=1).cpu()
                H = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                mean_probs += probs
                sq_probs += probs ** 2
                mean_entropy += H

        mean_probs /= num_samples
        sq_probs /= num_samples
        mean_entropy /= num_samples
        variance = sq_probs - mean_probs ** 2
        return mean_probs, variance, mean_entropy

    def get_logits_and_probs(self, data, device):
        """Runs a single full-graph forward pass to extract raw logits."""
        self.eval()
        data = data.to(device)
        with torch.no_grad():
            all_logits = self.forward(data.x, data.edge_index, data.iso_idx).cpu()
        probs = F.softmax(all_logits, dim=1)
        return all_logits, probs


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
