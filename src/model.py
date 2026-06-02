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
        hidden_dim=256,
        embed_dim=8,
        dropout_rate=0.3,
    ):
        super(CO2AssignmentGNN, self).__init__()
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

    def mc_dropout_predict(self, loader, device, num_nodes, num_samples=30):
        """Runs stochastic forward passes using optimal batch-first GPU execution."""
        self.train()  # Force dropout ON

        # Look up the number of output classes dynamically from the final linear layer
        num_classes = self.head[-1].out_features

        # We allocate these ONLY ONCE on the CPU to prevent memory fragmentation
        mean_probs = torch.zeros(
            (num_nodes, num_classes), dtype=torch.float32, device="cpu"
        )
        variance = torch.zeros(
            (num_nodes, num_classes), dtype=torch.float32, device="cpu"
        )
        # Mean per-sample entropy across MC passes — needed for BALD
        mean_entropy = torch.zeros(num_nodes, dtype=torch.float32, device="cpu")

        with torch.no_grad():
            from tqdm import tqdm

            for batch in tqdm(loader, desc="MC Dropout Inference (Batched)"):
                batch = batch.to(device)

                # Accumulators for this specific tiny batch (stays on GPU)
                batch_mean = None
                batch_sq = None
                batch_H = None  # mean per-sample entropy H(p_t)

                # Run the 30 stochastic passes on the exact same GPU subgraph
                for _ in range(num_samples):
                    logits = self.forward(batch.x, batch.edge_index, batch.iso_idx)
                    probs = F.softmax(logits[: batch.batch_size], dim=1)

                    # Per-sample entropy: H(p_t) = -sum_c p_t,c * log(p_t,c)
                    H_t = -(probs * torch.log(probs + 1e-10)).sum(dim=1)

                    if batch_mean is None:
                        batch_mean = probs.clone()
                        batch_sq = (probs**2).clone()
                        batch_H = H_t.clone()
                    else:
                        batch_mean += probs
                        batch_sq += probs**2
                        batch_H += H_t

                # Calculate batch statistics entirely on the GPU
                batch_mean /= num_samples
                batch_sq /= num_samples
                batch_H /= num_samples
                batch_var = batch_sq - (batch_mean**2)

                # Map the finished batch predictions back to the global CPU tensor just once
                target_n_ids = batch.n_id[: batch.batch_size].cpu()
                mean_probs[target_n_ids] = batch_mean.cpu()
                variance[target_n_ids] = batch_var.cpu()
                mean_entropy[target_n_ids] = batch_H.cpu()

        return mean_probs, variance, mean_entropy

    def get_logits_and_probs(self, loader, device, num_nodes):
        """Runs a standard evaluation pass to extract raw logits."""
        self.eval()  # Standard evaluation mode, NO dropout

        num_classes = self.head[-1].out_features
        all_logits = torch.zeros(
            (num_nodes, num_classes), dtype=torch.float32, device="cpu"
        )

        with torch.no_grad():
            from tqdm import tqdm

            for batch in tqdm(loader, desc="Extracting Logits"):
                batch = batch.to(device)
                logits = self.forward(batch.x, batch.edge_index, batch.iso_idx)

                target_n_ids = batch.n_id[: batch.batch_size].cpu()
                all_logits[target_n_ids] = logits[: batch.batch_size].cpu()

        # Calculate probabilities from logits
        probs = F.softmax(all_logits, dim=1)
        return all_logits, probs


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
