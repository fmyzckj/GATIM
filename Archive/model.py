import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphAttentionEncoder, Decoder, PointerNetwork
from torch.distributions.categorical import Categorical


class Encoder(nn.Module):
    def __init__(self, d_node, d_hid, n_head, dropout, alpha):
        super().__init__()
        self.dropout = dropout
        '''align features'''
        self.align = nn.Linear(1, d_node, bias=True)

        '''normalize features'''
        self.layer_norm = nn.LayerNorm(d_node, eps=1e-6)

        '''graph attention network'''
        self.attentions = [GraphAttentionEncoder(d_node, d_hid, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(n_head)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, ind_load, load, ind_generator, generator, adjacent):
        """align loads' feature vectors with those of generators"""
        load = self.align(load)  # *d
        x = torch.zeros([load.shape[0] + generator.shape[0], generator.shape[1]])
        ind_load = ind_load.flatten()  # 将 ind_load 调整为一维张量
        x[ind_load, :] = load
        ind_generator = ind_generator.flatten()  # 将 ind_generator 调整为一维张量
        generator = generator.to(torch.float)
        x[ind_generator, :] = generator
        x = self.layer_norm(x)  # normalize x
        '''encode via graph attention network'''
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adjacent) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class Actor(nn.Module):
    def __init__(self, n_head, n_layer, n_node, d_node, n_agent, d_k, d_v, d_hid, dropout):
        super().__init__()
        self.n_node = n_node
        self.n_agent = n_agent
        '''normalize features'''
        self.layer_norm = nn.LayerNorm(d_node, eps=1e-6)

        '''cross-attention between current agent and neighbor ones'''
        self.attention1_stack = nn.ModuleList([
            Decoder(n_head, d_node, d_k, d_v, d_hid, dropout)
            for _ in range(n_layer)])

        '''mask for sectionalization'''
        self.linear1 = nn.Linear(d_node, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        '''cross-attention between sectionalization and current path'''
        self.attention2_stack = nn.ModuleList([
            Decoder(n_head, d_node, d_k, d_v, d_hid, dropout)
            for _ in range(n_layer)])

        '''choose next node with pointer network'''
        self.ptr = PointerNetwork(1, d_node, d_k, d_v)

        '''get action'''
        self.linear2 = nn.Linear(d_node, 1, bias=False)

    def forward(self, obs, action):
        n, n_agent = self.n_node, self.n_agent
        obs = obs.unsqueeze(0)  # add a dimension as batch
        x_fon = torch.index_select(obs, 1, torch.arange(0, n))  # first n rows of obs
        y_fon = torch.index_select(obs, 1, torch.arange(n, n*n_agent))  # (n+1)~(n*n_agent) rows of obs
        x_path = torch.index_select(obs, 1, torch.arange(n*n_agent, obs.shape[1]))  # last n rows of obs

        '''cross-attention between current agent and neighbor ones'''
        for attention1 in self.attention1_stack:
            x_fon, _ = attention1(x_fon, y_fon, y_fon, mask=None)  # input without normalization

        '''learn mask for sectionalization'''
        mask_sec = self.linear1(x_fon)  # 0*n*1
        ind_nonzero = torch.nonzero(mask_sec != 0)[:, 1]  # indexes of candidate nodes

        if ind_nonzero.numel() == 0:
            a = 0

        mask_sec = self.sigmoid(mask_sec)  # zero nodes become 0.5 after sigmoid
        mask_sec[mask_sec <= 0.5] = torch.zeros([1])
        mask_sec[mask_sec > 0.5] = torch.ones([1])

        if (mask_sec.squeeze() == 0).all():  # repair the mask to have at least one candidate node, if masked all nodes
            ind_nonzero = ind_nonzero[torch.randperm(len(ind_nonzero))[0]]
            mask_sec[0][ind_nonzero] = torch.ones([1])  # keep to have at least one candidate node

        x_fon = x_fon * mask_sec.repeat(1, 1, x_fon.shape[2])  # 0*n*d

        '''cross-attention between candidate actions and current path'''
        # x_path = self.layer_norm(x_path)  # input with normalization
        # for attention2 in self.attention2_stack:
        #     x_fon, _ = attention2(x_fon, x_path, x_path, mask=None)
        # x = self.linear2(x_fon).squeeze(0, 2)  # actions' probabilities

        '''choose next node with pointer network'''
        x_path = self.layer_norm(x_path)  # input with normalization
        attn = self.ptr(x_fon, x_path, x_path, mask=None)  # actions' probabilities
        x = attn.squeeze(0)  # actions' probabilities

        '''get action'''
        x = x * mask_sec.squeeze(0, 2)  # mask unrelated nodes
        probs = Categorical(probs=x)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy()


class Critic(nn.Module):
    def __init__(self, d_node, n_node, n_agent):
        super().__init__()
        self.w_0 = nn.Linear(d_node, 1)
        self.w_1 = nn.Linear(n_node * (n_agent + 1), 64)
        self.w_2 = nn.Linear(64, 64)
        self.w_3 = nn.Linear(64, 1)
        self.tan_h = nn.Tanh()

    def forward(self, obs):
        x = self.w_0(obs)  # transform observation matrix (*d) to vector (*1)
        x = self.w_1(x.squeeze(1))  # x.squeeze(1): transform n*1 to n
        x = self.tan_h(x)
        x = self.w_2(x)
        x = self.tan_h(x)
        x = self.w_3(x)

        return x
