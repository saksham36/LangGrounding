import numpy as np
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
import random
from memory import Replay

class BootDQNAlgo():
    """The DQN algorithm."""

    def __init__(self, envs, model, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01,
                 max_grad_norm=0.5, recurrence=4,
                 preprocess_obss=None, reshape_reward=None, num_ensemble=2,
                 mask_prob=0.9, noise_scale=0, batch_size=100, replay_memory_size=100000,
                 min_replay_size=10, sgd_period=10):

        num_frames_per_proc = num_frames_per_proc or 8
        
        # Store parameters
        self.envs = envs
        self.model = model
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss

        self.num_ensemble = num_ensemble
        self.mask = torch.ones(num_ensemble, device=self.device)
        self.mask_prob = mask_prob
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.min_replay_size = min_replay_size
        self.sgd_period = sgd_period

        # Control parameters

        assert self.model.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.model.to(self.device)
        self.model.train()

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_ensemble)

        if self.model.recurrent:
            self.memory = torch.zeros(shape[1], self.model.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.model.memory_size, device=self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.loss_func = torch.nn.MSELoss()
        self.buffer = Replay(replay_memory_size)

    def store_transition(self, data):
        self.buffer.add(data)

    def select_action(self, obs, Q_idx):
        preprocessed_obs = self.preprocess_obss([obs], device=self.device)
        
        if self.model.recurrent:
            # try:
            Q, memory = self.model(preprocessed_obs, self.memory * self.mask.unsqueeze(1), Q_idx)
            # except Exception as e:
            #     print(e)
            #     import pdb; pdb.set_trace()
        else:
            Q = self.model(preprocessed_obs, Q_idx)
        action = np.random.choice(np.flatnonzero(Q == Q.max()))
        return int(action)

    def update(self, state, action, reward, state_, timestep):
        # Generate bootstrapping mask & reward noise
        mask = tuple(np.random.binomial(1, self.mask_prob, self.num_ensemble))
        noise = tuple(np.random.randn(self.num_ensemble))
        # Make transition and add to replay.
        transition = [
        state,
        action,
        float(reward),
        self.discount,
        state_,
        mask,
        noise,
        ]
        
        self.buffer.add(transition)

        if self.buffer.size < self.min_replay_size:
            return
        
        logs = dict()
        # Periodically sample from replay and do SGD for the whole ensemble.
        if timestep % self.sgd_period == 0:
            transitions = self.buffer.sample(self.batch_size)
            o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions
            for k in range(self.num_ensemble):
                transitions = [o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t]
                logs[k] = self.sgd_step(transitions, k)
            
        return logs

    def sgd_step(self, transitions, Q_idx):
        """Does a step of SGD for the whole ensemble over `transitions`."""
        # Q-learning loss with added reward noise + half-in bootstrap

        o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions
        o_tm1 = self.preprocess_obss(o_tm1, device=self.device)
        o_t = self.preprocess_obss(o_t, device=self.device)
        r_t += self.noise_scale * z_t[:,Q_idx]
        d_t = torch.Tensor(d_t, device=self.device)
        r_t = torch.Tensor(r_t, device=self.device)
        m_t = torch.Tensor(m_t, device=self.device)
        a_tm1 = torch.LongTensor(a_tm1, device=self.device)
        q_target, _ = self.model(o_t, self.memory * self.mask.unsqueeze(1), Q_idx)
        q_eval, _ = self.model(o_tm1, self.memory * self.mask.unsqueeze(1), Q_idx)
        q_eval = torch.gather(q_eval, 0, a_tm1.unsqueeze(1)).squeeze()
        td_error = m_t[:,Q_idx] * self.loss_func(q_eval, r_t + d_t * torch.amax(q_target,1))
        td_error = torch.sum(td_error)/torch.sum(m_t[:,Q_idx])
        self.optimizer.zero_grad()
        td_error.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.model.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {"loss": td_error}


        # for i in range(self.recurrence):
            
        #     sb = exps[inds + i]

        #     # Compute loss

        #     if self.acmodel.recurrent:
        #         Q, memory = self.acmodel(sb.obs, memory * sb.mask)
        #     else:
        #         Q = self.acmodel(sb.obs)
        #     value = Q[torch.max(Q,1)[1]
        #     # entropy = dist.entropy().mean()

        #     # policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

        #     value_loss = (value - sb.returnn).pow(2).mean()

        #     # loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
        #     loss = value_loss
        #     # Update batch values

        #     # update_entropy += entropy.item()
        #     update_value += value.mean().item()
        #     # update_policy_loss += policy_loss.item()
        #     update_value_loss += value_loss.item()
        #     update_loss += loss

        # # Update update values

        # # update_entropy /= self.recurrence
        # update_value /= self.recurrence
        # # update_policy_loss /= self.recurrence
        # update_value_loss /= self.recurrence
        # update_loss /= self.recurrence

        # # Update Q

        # self.optimizer.zero_grad()
        # update_loss.backward()
        # update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        # torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        # self.optimizer.step()

        # # Log some values

        # logs = {
        #     # "entropy": update_entropy,
        #     "value": update_value,
        #     # "policy_loss": update_policy_loss,
        #     "value_loss": update_value_loss,
        #     "grad_norm": update_grad_norm
        # }

        # return logs


class DQNAlgo(BaseAlgo):
    """The DQN algorithm."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None,epsilon=0.1):
        num_frames_per_proc = num_frames_per_proc or 8
        import pdb; pdb.set_trace()
        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)
        self.eps = epsilon
        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    Q, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    Q = self.acmodel(preprocessed_obs)
            
            max_action = torch.max(Q,1)[1]
            q_eval = Q[max_action]

            prob = torch.rand(max_action.shape)
            action = max_action # max_action will also get modified
            for eps_idx in range(prob.shape[0]): # iterating through env
                if prob[eps_idx] < self.eps:
                    action[eps_idx] = random.randint(0,self.env.envs[0].action_space.n -1)
            
            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            
            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = torch.amax(q_eval,1)
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences
        import pdb; pdb.set_trace()
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                Q_next, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                Q_next = self.acmodel(preprocessed_obs)
            next_value = Q_next[torch.max(Q_next,1)[1]]
        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    def update_parameters(self, exps):
        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize update values

        # update_entropy = 0
        update_value = 0
        # update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        # Initialize memory

        if self.acmodel.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Compute loss

            if self.acmodel.recurrent:
                Q, memory = self.acmodel(sb.obs, memory * sb.mask)
            else:
                Q = self.acmodel(sb.obs)
            value = Q[torch.max(Q,1)[1]]
            # entropy = dist.entropy().mean()

            # policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

            value_loss = (value - sb.returnn).pow(2).mean()

            # loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
            loss = value_loss
            # Update batch values

            # update_entropy += entropy.item()
            update_value += value.mean().item()
            # update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_loss += loss

        # Update update values

        # update_entropy /= self.recurrence
        update_value /= self.recurrence
        # update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        # Update Q

        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs = {
            # "entropy": update_entropy,
            "value": update_value,
            # "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.
        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.
        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = np.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
