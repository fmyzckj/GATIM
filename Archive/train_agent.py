import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from model import Actor, Critic
from methods import initialize_weights, Environment
from data import case_118

"""settings"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--n_episode", type=int, default=10000,  # number of episodes
                        help="total timesteps of the experiments")
    parser.add_argument("--num-steps", type=int, default=20,  # number of steps
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num-minibatches", type=int, default=4,  # mini batch size
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,  # PPO epochs
                        help="the K epochs to update the policy")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument("--env-id", type=str, default="PowerRestoration",
                        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")

    # Task specific arguments
    parser.add_argument('--num_agent', type=int, default=8,  # number of agents
                        help='dimension of key')
    parser.add_argument('--dim_k', type=int, default=64,
                        help='dimension of key')
    parser.add_argument('--dim_v', type=int, default=64,
                        help='dimension of value')
    parser.add_argument('--hidden', type=int, default=64,
                        help='inner dimension of FFN')
    parser.add_argument('--n_head', type=int, default=8,
                        help='number of head attentions.')
    parser.add_argument('--n_layer', type=int, default=6,
                        help='number of decoder layers.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    args.batch_size = int(args.num_agent * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


"""data"""
ind_load, load, ind_generator, generator, adjacent, load_weight = case_118()
feature = torch.load('embed_feature.txt')  # embedded feature matrix
bs = torch.tensor([[0], [26], [39], [54], [61], [86], [90], [109]])  # indexes of BSs
nbs = torch.tensor([[9], [11], [24], [25], [48], [53], [58], [60],
                    [64], [65], [68], [79], [88], [99], [102], [110]])  # indexes of NBSs

"""environment"""
envs = Environment(feature, adjacent, bs, nbs, ind_load, load, ind_generator, generator, load_weight)

"""main"""
"""based on the PPO code from CleanRL, https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py"""
if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    '''model and optimizer'''
    actor = Actor(n_head=args.n_head,
                  n_layer=args.n_layer,
                  n_node=adjacent.shape[0],
                  d_node=feature.shape[1]+1,  # feature dimensionality + capacity
                  n_agent=args.num_agent,
                  d_k=args.dim_k,
                  d_v=args.dim_v,
                  d_hid=args.hidden,
                  dropout=args.dropout).to(device)
    actor.apply(initialize_weights)
    optimizer_actor = optim.Adam(actor.parameters(), lr=args.learning_rate, eps=1e-5)

    critic = Critic(d_node=feature.shape[1]+1, n_node=feature.shape[0], n_agent=args.num_agent).to(device)
    critic.apply(initialize_weights)
    optimizer_critic = optim.Adam(critic.parameters(), lr=args.learning_rate, eps=1e-5)

    '''storage setup'''
    obs = torch.zeros((args.num_agent, args.num_steps) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_agent, args.num_steps) + envs.single_action_space.shape, dtype=torch.int).to(device)
    logprobs = torch.zeros(args.num_agent, args.num_steps).to(device)
    rewards = torch.zeros(args.num_agent, args.num_steps).to(device)
    dones = torch.zeros(args.num_agent, args.num_steps).to(device)
    values = torch.zeros(args.num_agent, args.num_steps).to(device)

    '''initialization'''
    global_step = 0
    start_time = time.time()
    next_obs = torch.zeros((args.num_agent, ) + envs.single_observation_space.shape).to(device)
    next_done = torch.zeros(args.num_agent).to(device)

    '''main loop'''
    for update in range(1, args.n_episode + 1):  # for each episode
        if args.anneal_lr:  # Annealing the rate if instructed to do so.
            frac = 1.0 - (update - 1.0) / args.n_episode
            lr_now = frac * args.learning_rate
            optimizer_actor.param_groups[0]["lr"] = lr_now
            optimizer_critic.param_groups[0]["lr"] = lr_now

        for step in range(0, args.num_steps):  # for each time step
            global_step += 1 * args.num_agent

            for agent in range(args.num_agent):  # for each agent
                # get current observation
                if next_done[agent] == torch.zeros(1):  # if not done
                    if step == 0 and agent == 0:
                        curr_ob, done = envs.get_obs(path=None, step=step, agent=agent)
                    else:
                        curr_ob, done = envs.get_obs(path=actions, step=step, agent=agent)
                    next_obs[agent] = torch.Tensor(curr_ob).to(device)
                    next_done[agent] = torch.Tensor(done).to(device)
                    obs[agent][step] = next_obs[agent]
                    dones[agent][step] = next_done[agent]
                else:
                    break

                # get action and it's value
                with torch.no_grad():
                    action, log_prob, _ = actor(next_obs[agent], action=None)
                    value = critic(next_obs[agent])
                    values[agent][step] = value.flatten()
                    actions[agent][step] = action
                    logprobs[agent][step] = log_prob

                # get action's reward
                reward = envs.get_reward(path=actions, agent=agent, step=step)
                rewards[agent][step] = torch.tensor(reward).to(device).view(-1)

        # bootstrap value if not done
        with torch.no_grad():
            returns = torch.zeros(args.num_agent, args.num_steps)
            for agent in range(args.num_agent):  # for each agent
                next_value = critic(next_obs[agent]).reshape(1, -1)  # add a dimension to value, for this agent only
                advantages = torch.zeros_like(rewards[agent]).to(device)  # 1*num_steps, for this agent only
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:  # if the last step
                        nextnonterminal = 1.0 - next_done[agent]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[agent][t + 1]
                    delta = rewards[agent][t] + args.gamma * nextvalues * nextnonterminal - values[agent][t]
                    # for this agent only
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    # for this agent only
                returns[agent] = advantages + values[agent]

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)  # batch_size = num_steps
        clipfracs = []
        for epoch in range(args.update_epochs):  # for each PPO epoch
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]  # choose an interval from an episode

                # off-policy
                newlogprob = torch.zeros(len(mb_inds))
                entropy = torch.zeros(len(mb_inds))
                newvalue = torch.zeros(len(mb_inds))
                for mb in range(len(mb_inds)):
                    _, newlogprob[mb], entropy[mb] = actor(b_obs[mb_inds[mb]], b_actions.long()[mb_inds[mb]])
                    newvalue[mb] = critic(b_obs[mb_inds[mb]])

                # KL between on-policy and off-policy
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer_actor.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                optimizer_actor.step()

                optimizer_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                optimizer_critic.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer_actor.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    writer.close()
