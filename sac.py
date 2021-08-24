import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
from copy import deepcopy


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.num_deploy = 0

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        
        # Now comes our deploy part
        self.deploy_policy = deepcopy(self.policy)
        
        # Now we find the criterias for switching policy
        self.only_reset_criteria = 'reset' in args.switching
        self.info_matrix_criteria = 'info' in args.switching
        self.need_feature = args.record_feature_sim
        self.need_kl = args.record_kl
        self.last_update = 0
        self.trigger_force_deploy = 0
        self.feature_reject, self.kl_reject = 0, 0
        self.force_deploy_interval = args.force_deploy_interval
        if self.force_deploy_interval < 0:
            self.force_deploy_interval = int(1e9)
        print('reset: {} | need f: {} | need kl: {}'.format(self.only_reset_criteria, self.need_feature, self.need_kl))
        max_power = np.log(1e9) / np.log(args.visit_eta)
        self.visit_deploy_T = (args.visit_eta ** np.arange(max_power)).astype(np.long)

    def select_action(self, state, explore=False, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            if explore:
                with torch.no_grad():
                    action, _, _ = self.deploy_policy.sample(state)
            else:
                action, _, _ = self.policy.sample(state)
        else:
            with torch.no_grad():
                _, _, action = self.deploy_policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def deploy(self, args, memory, hash_table, metrics, T, done=False, episode_steps=None):
        if self.only_reset_criteria and not done:
            return
        if self.info_matrix_criteria:
            if episode_steps % args.info_matrix_interval != 0 and not done:
                return
        
        # Compute something which might be use later
        states, actions, rewards, next_states, masks = memory.sample(batch_size=args.deploy_batch_size)
        states = torch.FloatTensor(states).to(self.device)
        if self.need_kl:
            with torch.no_grad():
                online_pd = self.policy.prob_distribution(states)
                deploy_pd = self.deploy_policy.prob_distribution(states)
            kl = torch.distributions.kl.kl_divergence(online_pd, deploy_pd).mean().item()
        if self.need_feature:
            with torch.no_grad():
                online_feature = self.policy.extract_feature(states).detach()
                deploy_feature = self.deploy_policy.extract_feature(states).detach()
            noline_feature_n = F.normalize(online_feature)
            deploy_feature_n = F.normalize(deploy_feature)
            sim = (noline_feature_n *deploy_feature_n).sum(-1).mean().item()
        
        # Now use some metric to decide deploying
        to_reset = False
        if args.switching in ['none', 'reset']:
            to_reset = True
        elif args.switching == 'fix':
            to_reset = T - self.last_update >= args.fix_interval
        elif args.switching == 'kl':
            if kl > args.policy_kl:
                to_reset = True
            else:
                self.kl_reject += 1
        elif args.switching in ['feature', 'reset+feature']:
            if sim < args.feature_sim:
                to_reset = True
            else:
                self.feature_reject += 1
        elif args.switching == 'visited':
            if hash_table.state_action_count in self.visit_deploy_T:
                to_reset = True
        elif args.switching == 'info-matrix':
            _, to_deploy = hash_table.info_matrix_value
            if to_deploy:
                to_reset = True
        elif args.switching == 'info-det':
            _, to_deploy = hash_table.info_matrix_det
            if to_deploy:
                to_reset = True
        else:
            raise NotImplementedError('Your switching policy is not implemented due to some issues')

        if not to_reset and T - self.last_update > self.force_deploy_interval and self.force_deploy_interval > 0:
            to_reset = True
            self.trigger_force_deploy += 1
        if to_reset:
            self.deploy_policy.load_state_dict(self.policy.state_dict())
            self.last_update = T
            self.num_deploy += 1
            metrics['switch T'].append(T)
        if 'force trigger' not in metrics:
            metrics['force trigger'] = []
        metrics['force trigger'].append(self.trigger_force_deploy)

        if args.record_feature_sim:
            if 'feature try T' not in metrics:
                metrics['feature try T'] = []
                metrics['feature reject'] = []
                metrics['feature sim'] = []
            metrics['feature try T'].append(T)
            metrics['feature reject'].append(self.feature_reject)
            metrics['feature sim'].append(sim)
        
        if args.record_kl:
            if 'kl try T' not in metrics:
                metrics['kl try T'] = []
                metrics['kl reject'] = []
                metrics['kl when reset'] = []
                metrics['kl'] = []
            metrics['kl try T'].append(T)
            metrics['kl reject'].append(self.kl_reject)
            metrics['kl'].append(kl)
            if to_reset:
                metrics['kl when reset'].append(kl)
    
    # Save model parameters
    def save_model(self, save_dir, T=None):
        if T is None:
            raise NotImplementedError
        else:   
            print('Saving models at itereration {}'.format(T))
        torch.save(self.policy.state_dict(), os.path.join(save_dir, 'online_act_{}.pth'.format(T)))
        torch.save(self.deploy_policy.state_dict(), os.path.join(save_dir, 'deploy_act_{}.pth'.format(T)))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, 'critic_{}.pth'.format(T)))

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

