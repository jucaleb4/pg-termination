# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
import multiprocessing as mp

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from pg_termination.logger import BasicLogger

def make_env(env_name, max_episode_steps):
    def thunk():
        if max_episode_steps > 0:
            env = gym.make(env_name, max_episode_steps=max_episode_steps)
        else:
            env = gym.make(env_name)
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, linear_only=False):
        super().__init__()
        if linear_only:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                layer_init(nn.Linear(64, 64)),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                layer_init(nn.Linear(64, 64)),
                layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
            )
            return

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def _train(settings):
    settings["num_envs"] = 1
    settings["torch_deterministic"] = True

    # to be filled in runtime
    """the batch size (computed in runtime)"""
    settings["ppo_batch_size"] = int(settings["num_envs"] * settings["ppo_rollout_len"])
    """the mini-batch size (computed in runtime)"""
    settings["ppo_minibatch_size"] = int(settings["ppo_batch_size"] // settings["ppo_num_minibatches"])
    """the number of iterations (computed in runtime)"""
    settings["num_iterations"] = settings["max_obs"] // settings["ppo_batch_size"]

    # TRY NOT TO MODIFY: seeding
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.backends.cudnn.deterministic = settings["torch_deterministic"]

    device = torch.device("cuda" if torch.cuda.is_available() and settings["ppo_cuda"] else "cpu")

    # logger
    logger_validation = BasicLogger(
        fname=os.path.join(settings["log_folder"], "validation_seed=%d.csv" % settings['seed']), 
        keys=["value", "opt_lb", "uni_opt_lb", "true value", "true opt_lb", "true uni_opt_lb"],
        dtypes=['f'] * 6,
    )
    logger_ep = BasicLogger(
        fname=os.path.join(settings["log_folder"], "ep_cost_seed=%d.csv" % settings['seed']),  
        keys=["ep_cost", "ep_len", 'cum_samps'],
        dtypes=['f', 'd', 'd'],
    )

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(settings["env_name"], settings["gym_max_episode_steps"]) for i in range(settings["num_envs"])],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, settings["ppo_linear_only"]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=settings["ppo_lr"], eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((settings["ppo_rollout_len"], settings["num_envs"]) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((settings["ppo_rollout_len"], settings["num_envs"]) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((settings["ppo_rollout_len"], settings["num_envs"])).to(device)
    rewards = torch.zeros((settings["ppo_rollout_len"], settings["num_envs"])).to(device)
    dones = torch.zeros((settings["ppo_rollout_len"], settings["num_envs"])).to(device)
    values = torch.zeros((settings["ppo_rollout_len"], settings["num_envs"])).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=settings["seed"])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(settings["num_envs"]).to(device)

    s_time = time.time()
    for iteration in range(1, settings["num_iterations"] + 1):
        # Annealing the rate if instructed to do so.
        if settings["ppo_anneal_lr"]:
            frac = 1.0 - (iteration - 1.0) / settings["num_iterations"]
            lrnow = frac * settings["ppo_lr"]
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, settings["ppo_rollout_len"]):
            global_step += settings["num_envs"]
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # TODO: Adapt to multiple envs
            if next_done and "episode" in infos:
                info = infos
                # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                # logger_ep.log(ep_cost, ep_len, cum_samps)
                # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                t = iteration
                if ((t) <= 100 and (t) % 5 == 0) or (t) % 100==0 or t<=10:
                    print("[%d] charts/episodic_return" % t, info["episode"]["r"], global_step)
                    # print("[%d] charts/episodic_length" % t, info["episode"]["l"], global_step)

                # TODO: Can we always access [0]
                logger_ep.log(info["episode"]["r"][0], info["episode"]["l"][0], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(settings["ppo_rollout_len"])):
                if t == settings["ppo_rollout_len"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + settings["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + settings["gamma"] * settings["ppo_gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(settings["ppo_batch_size"])
        clipfracs = []
        for epoch in range(settings["ppo_update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, settings["ppo_batch_size"], settings["ppo_minibatch_size"]):
                end = start + settings["ppo_minibatch_size"]
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > settings["ppo_clip_coef"]).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if settings["ppo_norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - settings["ppo_clip_coef"], 1 + settings["ppo_clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if settings["ppo_clip_vloss"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -settings["ppo_clip_coef"],
                        settings["ppo_clip_coef"],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - settings["ppo_ent_coef"] * entropy_loss + v_loss * settings["ppo_vf_coef"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), settings["ppo_max_grad_norm"])
                optimizer.step()

            if settings["ppo_target_kl"] is not None and approx_kl > settings["ppo_target_kl"]:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("charts/ppo_lr", optimizer.param_groups[0]["ppo_lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if time.time() - s_time > settings["max_runtime_in_sec"]:
            print("=== Breaking early because we exceeded the max runtime ===")
            break

    # validation
    validation_time = settings["max_runtime_in_sec"]/2
    V = estimate_random_reset_value(envs, agent, device, settings["validation_k"], validation_time/2)
    true_V = np.inf
    V_lb = uni_V_lb = true_V_lb = true_uni_V_lb = -np.inf
    logger_validation.log(V, V_lb, uni_V_lb, true_V, true_V_lb, true_uni_V_lb)

    envs.close()
    logger_ep.save(max_size=10_000)
    logger_validation.save(max_size=10_000)
    # writer.close()

def estimate_random_reset_value(envs, agent, device, n_replicates=30, time_limit=np.inf):
        """
        Estimates the quantity

        $$
            E_(s ~ rho)[ E [ c_0 + gamma*c_1 + gamma**2 * c_2 + ... | s_0=s] ],
        $$
        
        where $rho$ is the reset distribution. This requires the environment to
        be have a termination condition in finite time to ensure non-infinite
        runtime. A runtime can be supplied.

        :param get_act: policy 
        :param n_replicates: how many Monte Carlo replicates to estimate
        :param time_limit: maximum runtime to run for
        :return V_est: estimate of randomly reset value function
        """
        s_time = time.time()

        # run until reset
        terminate = 0
        next_obs = torch.Tensor(envs.observation_space.sample()).to(device)
        while (not terminate):
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            if time.time() - s_time > time_limit:
                break

            terminate = np.logical_or(terminations, truncations)
            next_obs = torch.Tensor(next_obs).to(device)

        t = 0
        replic_id = 0
        curr_V = 0
        V_est = 0
        while replic_id < n_replicates:
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            terminate = np.logical_or(terminations, truncations)
            next_obs  = torch.Tensor(next_obs).to(device)

            curr_V += reward
            t += 1
            if terminate:
                replic_id += 1
                alpha = 1./replic_id
                V_est = (1.-alpha) * V_est + alpha*curr_V
                t = curr_V = 0
            if time.time() - s_time > time_limit:
                break

        return V_est[0] if replic_id > 0 else np.inf

def train(settings):
    seed_0 = settings["seed_0"]
    n_seeds = settings["n_seeds"]
    parallel = settings["parallel"]

    num_workers = 1
    try:
        num_workers = min(n_seeds, len(os.sched_getaffinity(0))-1)
        print("Parallel PO experiements with %d workers (%d jobs, %d max cpu)" % (
            num_workers, n_seeds, mp.cpu_count())
        )
    except:
        if 'sched_getffinity' not in dir(os):
            print("Function `os.sched_getaffinity(0)` not available on current OS.\nSetting parallel=False.\nSee https://stackoverflow.com/questions/42658331/python-3-on-macos-how-to-set-process-affinity")
            parallel = False

    worker_queue = []

    for seed in range(seed_0, seed_0+n_seeds):
        customized_settings = settings.copy()
        customized_settings["seed"] = seed

        if not parallel:
            _train(customized_settings)
            # skips line below
            continue

        # avoid over-assigning CPUs
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if len(worker_queue) == num_workers:
            # wait for all workers to finish
            for p in worker_queue:
                p.join()
            worker_queue = []
        p = mp.Process(target=_train, args=(customized_settings,))
        p.start()
        worker_queue.append(p)