import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGRUEmbedder(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=32, num_classes=10):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(32)
        self.gru = nn.GRU(32, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        h = F.relu(self.bn(self.conv(x)))
        h = h.permute(0, 2, 1)
        out, _ = self.gru(h)
        emb = out.mean(dim=1)
        logits = self.head(emb)
        return emb, logits
