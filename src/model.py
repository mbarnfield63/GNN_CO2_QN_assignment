import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

# Full-batch forward passes over the ~2.5M-node graph OOM on both 8GB GPUs and
# large CPU RAM (message-passing activations scale with edge count). All
# inference below goes through mini-batches instead.
NUM_NEIGHBORS = [15, 10]
INFERENCE_BATCH_SIZE = 2048


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

    def _batched_forward(self, data, device):
        """One inference pass over every node, in NeighborLoader mini-batches.

        Respects self.training (dropout stays on if the caller left the model
        in train() mode, e.g. for MC Dropout sampling).
        """
        loader = NeighborLoader(
            data, num_neighbors=NUM_NEIGHBORS, batch_size=INFERENCE_BATCH_SIZE, shuffle=False
        )
        num_classes = self.head[-1].out_features
        out = torch.zeros((data.num_nodes, num_classes), dtype=torch.float32)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = self.forward(batch.x, batch.edge_index, batch.iso_idx)
                out[batch.n_id[: batch.batch_size].cpu()] = logits[: batch.batch_size].cpu()
        return out

    def mc_dropout_predict(self, data, device, num_samples=30):
        """Runs stochastic forward passes over the full graph (mini-batched)."""
        self.train()  # Force dropout ON
        num_classes = self.head[-1].out_features
        num_nodes = data.num_nodes

        mean_probs = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
        sq_probs = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
        mean_entropy = torch.zeros(num_nodes, dtype=torch.float32)

        for _ in tqdm(range(num_samples), desc="MC Dropout Inference"):
            logits = self._batched_forward(data, device)
            probs = F.softmax(logits, dim=1)
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
        """Runs a full-graph forward pass (mini-batched) to extract raw logits."""
        self.eval()
        all_logits = self._batched_forward(data, device)
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
