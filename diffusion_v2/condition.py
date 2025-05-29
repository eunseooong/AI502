import torch.nn as nn

class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)

    def forward(self, y):
        return self.label_emb(y)
