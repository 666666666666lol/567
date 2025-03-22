from torch import nn
from torchvision.models import resnet18
import torch


class SelfAttention(nn.Module):
    """
    SelfAttention implements multi-head self-attention mechanism used in transformers.
    It computes attention weights and generates weighted combinations of values for each query.
    """

    def __init__(self, embed_dim, num_head, is_mask=True):
        super(SelfAttention, self).__init__()
        assert (
            embed_dim % num_head == 0
        )  # Ensure that embed_dim is divisible by num_head
        self.num_head = num_head
        self.is_mask = is_mask
        self.linear1 = nn.Linear(embed_dim, 3 * embed_dim)  # First linear layer
        self.linear2 = nn.Linear(embed_dim, embed_dim)  # Second linear layer

        self.dropout1 = nn.Dropout(0.1)  # Dropout layer after attention weights
        self.dropout2 = nn.Dropout(0.1)  # Dropout layer after final output

    def forward(self, x):
        """x shape: N, S, V"""
        x = self.linear1(x)  # Transform shape to N, S, 3V
        n, s, v = x.shape
        """Split into heads, shape becomes N, S, H, V"""
        x = x.reshape(n, s, self.num_head, -1)
        """Transpose dimensions, shape becomes N, H, S, V"""
        x = torch.transpose(x, 1, 2)
        """Split into Q, K, V"""
        query, key, value = torch.chunk(x, 3, -1)
        dk = value.shape[-1] ** 0.5
        """Compute self-attention"""
        w = torch.matmul(query, key.transpose(-1, -2)) / dk  # w shape: N, H, S, S
        # if self.is_mask:
        #     mask = torch.tril(torch.ones(w.shape[-1], w.shape[-1])).to(w.device)
        #     w = w * mask - 1e10 * (1 - mask)
        w = torch.softmax(w, dim=-1)  # Apply softmax normalization
        w = self.dropout1(w)
        attention = torch.matmul(
            w, value
        )  # Combine vectors based on attention weights, shape: N, H, S, V
        attention = attention.permute(0, 2, 1, 3)
        n, s, h, v = attention.shape
        attention = attention.reshape(n, s, h * v)
        return self.dropout2(
            self.linear2(attention)
        )  # Final output after linear transformation


class Block(nn.Module):
    """
    Block represents a transformer block with multi-head self-attention and feed-forward layers.
    It performs layer normalization, attention, and residual connections.
    """

    def __init__(self, embed_dim, num_head, is_mask):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, num_head, is_mask)
        self.ln_2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 6),
            nn.ReLU(),
            nn.Linear(embed_dim * 6, embed_dim),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """Compute multi-head self-attention"""
        attention = self.attention(self.ln_1(x))
        """Residual connection"""
        x = attention + x
        x = self.ln_2(x)
        """Compute feed-forward part"""
        h = self.feed_forward(x)
        h = self.dropout(h)
        x = h + x  # Add residual connection
        return x


class AbsPosEmb(nn.Module):
    """
    AbsPosEmb generates absolute positional embeddings for the input feature map.
    This is used to inject spatial positional information into the model.
    """

    def __init__(self, fmap_size, dim_head):
        super().__init__()
        height, width = fmap_size
        scale = dim_head**-0.5
        self.height = nn.Parameter(
            torch.randn(height, dim_head, dtype=torch.float32) * scale
        )
        self.width = nn.Parameter(
            torch.randn(width, dim_head, dtype=torch.float32) * scale
        )

    def forward(self):
        # Embedding shape: (height, width, dim)
        emb = self.height.unsqueeze(1) + self.width.unsqueeze(0)
        h, w, d = emb.shape
        emb = emb.reshape(h * w, d)  # Flatten to shape (h*w, dim)
        return emb


class OcrNet(nn.Module):
    """
    OcrNet is a deep neural network for Optical Character Recognition (OCR) on license plates.
    It uses a ResNet backbone for feature extraction and a transformer-like decoder for sequence modeling.
    """

    def __init__(self, num_class):
        super(OcrNet, self).__init__()
        resnet = resnet18(True)
        backbone = list(resnet.children())
        self.backbone = nn.Sequential(
            nn.BatchNorm2d(3),
            *backbone[:3],
            *backbone[4:8],
        )
        self.decoder = nn.Sequential(
            Block(512, 8, False),
            Block(512, 8, False),
        )
        self.out_layer = nn.Linear(512, num_class)
        self.abs_pos_emb = AbsPosEmb((3, 9), 512)

    def forward(self, x):
        """Forward pass through the network"""
        x = self.backbone(x)
        n, c, h, w = x.shape
        print(x.shape)
        x = x.permute(0, 3, 2, 1).reshape(n, w * h, c)
        x = x + self.abs_pos_emb()
        x = self.decoder(x)
        x = x.permute(1, 0, 2)
        y = self.out_layer(x)
        return y


if __name__ == "__main__":
    m = OcrNet(70)
    print(m)
    x = torch.randn(32, 3, 48, 144)
    print(m(x).shape)
