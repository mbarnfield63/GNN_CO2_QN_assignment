import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class CO2AssignmentGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        num_isotopes,
        num_classes,
        hidden_dim=128,
        embed_dim=8,
        dropout_rate=0.2,
    ):
        super(CO2AssignmentGNN, self).__init__()
        self.dropout_rate = dropout_rate

        # Structural embedding for isotope context
        self.iso_embed = nn.Embedding(num_isotopes, embed_dim)
        total_input_dim = input_dim + embed_dim

        # GraphSAGE scales efficiently without N^2 attention weights
        self.conv1 = SAGEConv(total_input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # Output head maps to the discrete combinatorial classes
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, edge_index, iso_idx):
        iso_emb = self.iso_embed(iso_idx)
        x = torch.cat([x, iso_emb], dim=1)

        x = self.conv1(x, edge_index)
        x = F.gelu(x)

        x = self.conv2(x, edge_index)
        x = F.gelu(x)

        return self.head(x)

    def mc_dropout_predict(self, x, edge_index, iso_idx, num_samples=50):
        """Runs stochastic forward passes using a memory-safe running average."""
        self.train()  # Force dropout ON

        mean_probs = None
        mean_sq_probs = None

        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x, edge_index, iso_idx)
                probs = F.softmax(logits, dim=1)

                # Running average to save massive amounts of RAM
                if mean_probs is None:
                    mean_probs = probs.clone()
                    mean_sq_probs = (probs**2).clone()
                else:
                    mean_probs += probs
                    mean_sq_probs += probs**2

        mean_probs /= num_samples
        mean_sq_probs /= num_samples

        # Variance = E[X^2] - (E[X])^2
        variance = mean_sq_probs - (mean_probs**2)

        return mean_probs, variance
