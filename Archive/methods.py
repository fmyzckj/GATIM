import numpy
import torch


class ScaledDotProductAttention(torch.nn.Module):  # Attention(Q, K, V) = softmax (QK.T/sqrt(d_k))V
    """Scaled Dot-Product Attention, from https://github.com/jadore801120/attention-is-all-you-need-pytorch"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)  # change the entries masked with True to -1e9

        attn = self.dropout(torch.nn.functional.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

# class ScaledDotProductAttention(torch.nn.Module):
#     """Scaled Dot-Product Attention, from https://github.com/jadore801120/attention-is-all-you-need-pytorch"""
#     def __init__(self, temperature, attn_dropout=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.dropout = torch.nn.Dropout(attn_dropout)
#
#     def forward(self, q, k, v, mask=None):
#         if len(k.shape) == 3:  # q, k, v equal to input, without mapping with w
#             mask_fon = torch.sum(q, dim=2)[0]  # mask of first order neighborhood, shape: [n_node]
#             mask_fon[mask_fon == 0] = torch.zeros([1])
#             mask_fon[mask_fon != 0] = torch.ones([1])
#             attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
#         else:
#             mask_fon = torch.sum(q, dim=3)[0][0]  # mask of first order neighborhood, shape: [n_node]
#             mask_fon[mask_fon == 0] = torch.zeros([1])
#             mask_fon[mask_fon != 0] = torch.ones([1])
#             attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
#
#         if mask is not None:
#             attn = attn.masked_fill(mask == 0, -1e9)
#
#         if len(k.shape) == 3:  # q, k, v equal to input, without mapping with w
#             attn = torch.sum(attn, dim=2)
#             attn = torch.nn.functional.softmax(attn, dim=-1)
#             attn = attn * mask_fon.unsqueeze(0)  # masked (with first order neighborhood) attention scores
#             output = []
#         else:
#             attn = self.dropout(torch.nn.functional.softmax(attn, dim=-1))
#             mask_fon = mask_fon.unsqueeze(0).unsqueeze(0).unsqueeze(3)  # shape: [1, 1, 1, n_node]
#             attn = attn * mask_fon.repeat(1, 8, 1, 826)  # masked (with first order neighborhood) attention scores
#             output = torch.matmul(attn, v)
#
#         return output, attn


def initialize_weights(m):
    """initialize model parameters"""
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        weight = m.weight.data  # 获取权重张量（添加）
        weight = weight.to(weight.dtype)  # 将权重类型转换为与原始权重类型相同的类型
        torch.nn.init.orthogonal_(weight, numpy.sqrt(2))  # 正交初始化
        # torch.nn.init.constant_(weight, 0.0)  # 保证初始值相同


class Environment:
    """power system restoration environment"""

    def __init__(self, feature, adjacent, bs, nbs, ind_load, load, ind_generator, generator, load_weight):
        super().__init__()
        '''define environment properties'''
        self.feature = feature  # n*d, embedded feature matrix from encoder
        self.adjacent = adjacent  # n*n, adjacent matrix
        self.bs = bs  # indexes of BSs
        self.num_agent = bs.shape[0]  # number of agents
        self.nbs = nbs  # indexes of NBSs
        self.ind_load = ind_load  # indexes of loads
        self.load = load  # loads' power demand
        self.ind_generator = ind_generator  # indexes of generators
        self.generator = generator  # ramping of generators
        self.load_weight = load_weight  # loads' importance

        self.single_action_space = torch.empty(1)  # dimension of an action
        self.single_observation_space = torch.empty([feature.shape[0] * (self.num_agent + 1), feature.shape[1] + 1])
        # dimension of an observation

    def get_obs(self, path, step, agent):  # all agents' current paths, current time step, this agent
        """get next observation by current action"""
        x, adjacent, bs, nbs, num_agent, ind_generator, generator = (
            self.feature, self.adjacent, self.bs, self.nbs, self.num_agent, self.ind_generator, self.generator)
        # bs: tensor, all agents' BS; nbs: tensor, all agents' NBSs

        '''add BS to the beginning of the path'''
        bs = bs.unsqueeze(1)  # expand to 3-D bs to match 3-D path
        if path is None:  # if at beginning without path
            path = bs
        else:
            path = torch.cat((bs, path), 1)  # add BS to the beginning of the path

        '''get this agent's generators' current capacity'''
        capacity = torch.zeros(1)
        bs = bs.squeeze(1)  # reduce to 2-D bs to match 2-D x
        capacity += generator[torch.eq(bs[agent], ind_generator.squeeze()), step]  # add bs capacity

        for i in range(nbs.shape[0]):
            if (path[agent] == nbs[i].unsqueeze(1)).any():  # if the i-th NBS on path
                nbs_step = torch.nonzero(torch.eq(path[agent], nbs[i].unsqueeze(1)).squeeze(1))
                # step of choosing the nbs
                capacity += generator[torch.eq(nbs[i], ind_generator).squeeze(), (step - nbs_step).squeeze()]
                # add nbs capacity

        capacity = capacity.repeat(x.shape[0]).unsqueeze(1)  # copy n*1 capacity, then expand to 2-D
        x = torch.cat((x, capacity), 1)  # n*(d+1), add capacity to feature matrix

        '''generate first-order networks (fon_x, fon_y) and current path network (x_path)'''
        x_fon = torch.zeros(x.size())  # n*(d+1)
        y_fon = torch.zeros(x.size())  # n*(d+1)
        temp = torch.zeros([1, x.shape[1]])  # 1*(d+1)

        for i in range(num_agent):
            logic = adjacent[path[i], :].sum(dim=0) > 0  # first order neighbor nodes of i-th agent
            logic = logic.squeeze(0)
            for j in range(num_agent):
                curr_path = path[j].squeeze()
                curr_path = curr_path[curr_path > 0]  # indexes of nodes on the path
                logic[curr_path] = False  # delete j-th agent's visited nodes

            if i == agent:  # if current agent
                x_fon[logic, :] = x[logic, :]
                if ~logic.any():  # if any logic is false (no candidate action)
                    done = torch.ones(1)  # restoration done
                else:
                    done = torch.zeros(1)
            else:
                y_fon[logic, :] = x[logic, :]
                temp = torch.cat((temp, y_fon), 0)  # all neighbor agents' observations
        ind = torch.arange(1, temp.shape[0])  # indexes of 1~end rows of temp
        y_fon = torch.index_select(temp, 0, ind)  # retain 1~end rows of temp

        x_path = torch.zeros(x.shape)  # n*(d+1)
        x_path[path[agent], :] = x[path[agent], :]

        return torch.cat((x_fon, y_fon, x_path), 0), done

    def get_reward(self, path, agent, step):
        x, adjacent, bs, nbs, num_agent, ind_generator, generator, ind_load, load, load_weights = (
            self.feature, self.adjacent, self.bs, self.nbs, self.num_agent, self.ind_generator, self.generator,
            self.ind_load, self.load, self.load_weight)
        # bs: tensor, all agents' BS; nbs: tensor, all agents' NBSs

        '''get this agent's generators' current capacity'''
        capacity = torch.zeros(1)
        bs = bs.squeeze(1)  # reduce to 2-D bs to match 2-D x
        capacity += generator[torch.eq(bs[agent], ind_generator.squeeze()), step]  # add bs capacity

        for i in range(nbs.shape[0]):
            if (path[agent] == nbs[i].unsqueeze(1)).any():  # if the i-th NBS on path
                nbs_step = torch.nonzero(torch.eq(path[agent], nbs[i].unsqueeze(1)).squeeze(1))
                # step of choosing the nbs
                capacity += generator[torch.eq(nbs[i], ind_generator).squeeze(), (step - nbs_step).squeeze()]
                # add nbs capacity

        '''calculate reward'''
        if (ind_load == path[agent][step]).any():  # if any load on path
            restored_load = torch.eq(ind_load, path[agent][step])  # logic of restored load in ind_load
            reward = torch.mul(load_weights[restored_load], load[restored_load])  # restored power

            curr_path = path[agent].squeeze()
            curr_path = curr_path[curr_path > 0]  # indexes of nodes on the path
            curr_demand = torch.zeros(1)
            for i in range(curr_path.numel()):
                if (curr_path[i] == ind_load).any():
                    curr_demand += load[torch.eq(curr_path[i], ind_load)]
            # 当前path的load demand 与目前发电机capacity的差值
            if capacity - curr_demand >= 0:  # if satisfy constraint
                cv = 0
            else:
                cv = - (capacity - curr_demand)  # >0
            reward = reward - cv
        else:
            # 当路径走到NBS时，奖励=NBS的capacity
            reward = - generator[torch.eq(ind_generator, path[agent][step]).squeeze(), 0]

        return reward
