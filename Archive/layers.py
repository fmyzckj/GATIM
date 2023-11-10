import torch
import torch.nn as nn
import torch.nn.functional as F

from methods import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """multi-head attention layer, from https://github.com/jadore801120/attention-is-all-you-need-pytorch"""

    def __init__(self, n_head, d_node, d_k, d_v, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_node, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_node, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_node, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_node, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_node, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionWiseFeedForward(nn.Module):
    """Two-feed-forward layer, from https://github.com/jadore801120/attention-is-all-you-need-pytorch"""

    def __init__(self, d_in, d_hid, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid, bias=False)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in, bias=False)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x


class PointerNetwork(nn.Module):
    """pointer network"""

    def __init__(self, n_head, d_node, d_k, d_v):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_node, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_node, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_node, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_node, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        # k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        # v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        _, attn = self.attention(q, k, v, mask)

        return attn


class GraphAttentionEncoder(nn.Module):
    """encoder with graph attention network, from https://github.com/Diego999/pyGAT"""

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        wh = torch.mm(h, self.W)  # feature augment
        e = self._prepare_attentional_mechanism_input(wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, wh):
        # wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # wh1&2.shape (N, 1)
        # e.shape (N, N)
        wh1 = torch.matmul(wh, self.a[:self.out_features, :])
        wh2 = torch.matmul(wh, self.a[self.out_features:, :])
        # broadcast add
        e = wh1 + wh2.T

        return self.leaky_relu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Decoder(nn.Module):
    """decoder with multi-head attention and feed forward network"""

    def __init__(self, n_head, d_node, d_k, d_v, d_hid, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, d_node, d_k, d_v, dropout)
        self.ffn = PositionWiseFeedForward(d_node, d_hid, dropout)

    def forward(self, q, k, v, mask=None):
        dec_output, dec_attn = self.attn(q, k, v, mask)
        dec_output = self.ffn(dec_output)

        return dec_output, dec_attn
