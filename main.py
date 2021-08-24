import argparse
import datetime
from genericpath import exists
import gym
import numpy as np
import itertools
import torch
import os
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from tqdm import tqdm
from hsh import HashTable

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--switching', default="none", type=str,
                    choices=['none', 'reset', 'fix', \
                             'kl', \
                             'feature', 'reset+feature', \
                             'visited', 'info-matrix', 'info-det'],
                    help='switching policy')
parser.add_argument('--fix_interval', default=1000, type=int, help='for fix policy')
parser.add_argument('--feature_sim', default=0.98, type=float, help='for feature-based policy')
parser.add_argument('--policy_kl', default=1, type=float, help='for kl-based policy')
parser.add_argument('--info-matrix-interval', default=10, type=int, help='info-matrix costs a lot')
parser.add_argument('--force-deploy-interval', default=-1, type=int, help='force deploy length')

parser.add_argument('--record-feature-sim', default=False, action='store_true')
parser.add_argument('--record-kl', default=False, action='store_true')
parser.add_argument('--mu-explore', default=False, action='store_true')
parser.add_argument('--random-T', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 2000)')
parser.add_argument('--id', default="default", type=str,
                    help='experiment name')

parser.add_argument('--hash-dim', default=32, type=int)
parser.add_argument('--count-bonus', default=0., type=float)
parser.add_argument('--visit-eta', default=2, type=float, help='deploy for all [eta^k]')

parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10000 steps (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.001, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--deploy_batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=2000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--num_episodes', type=int, default=1000001, metavar='N',
                    help='maximum number of episodes (default: 100000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=50, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=2000, metavar='N',
                    help='Steps before start training (default: 2000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--checkpoint_interval', type=int, default=500000, metavar='N',
                    help='checkpoint interval')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()


save_dir = os.path.join(os.getcwd(), 'results', args.id + '_seed_{}'.format(args.seed))
if os.path.exists(save_dir):
    r = input("Save dir [ {} ] already exists, press [Y] to continue...".format(save_dir))
    if r != 'Y':
        print('Aborted...')
        exit(0)
os.makedirs(save_dir, exist_ok=True)

metrics = {'iter': [], 'score': [], 'deploy': [], 'switch T': []}
if args.switching in ['feature', 'reset+feature']:
    args.record_feature_sim = True
if args.switching == 'kl':
    args.record_kl = True

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env, test_env = gym.make(args.env_name), gym.make(args.env_name)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

env.seed(args.seed)
env.action_space.seed(args.seed)
test_env.seed(args.seed + 1)
test_env.action_space.seed(args.seed + 1)


# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
hash_table = HashTable(args, env.observation_space.shape[0], env.action_space.shape[0])

#Tesnorboard
# writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
#                                                              args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
step_since_last_eval = 0
updates = 0

tqdm_bar = tqdm(range(1, 1 + args.num_steps))

for i_episode in range(1, 1 + args.num_episodes):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.random_T > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            # Sample action from policy
            if args.mu_explore:
                action = agent.select_action(state, evaluate=True)
            else:
                action = agent.select_action(state, explore=True) # Sample from deployed policy

        if len(memory) > args.start_steps and total_numsteps % args.updates_per_step == 0:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                '''
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                '''
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        
        info_index = episode_steps // args.info_matrix_interval
        state_count = hash_table.step(state, action, need_info_matrix=('info' in args.switching), info_index=info_index)
        
        episode_steps += 1
        total_numsteps += 1
        tqdm_bar.update(1)
        step_since_last_eval += 1
        episode_reward += reward
    
        if len(memory) > max(args.start_steps, args.random_T):
            agent.deploy(args, memory, hash_table, metrics, total_numsteps, done, episode_steps=episode_steps)

        if total_numsteps % args.checkpoint_interval == 0:
            agent.save_model(save_dir, total_numsteps)

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        new_reward = reward + args.count_bonus * (state_count ** -0.5)
        memory.push(state, action, new_reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    # writer.add_scalar('reward/train', episode_reward, i_episode)
    tqdm_bar.set_description("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if step_since_last_eval > 10000 and args.eval is True:
        step_since_last_eval = 0
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = test_env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = test_env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        metrics['iter'].append(total_numsteps)
        metrics['score'].append(avg_reward)
        metrics['deploy'].append(agent.num_deploy)
        torch.save(metrics, os.path.join(save_dir, 'metrics.pth'))
        if total_numsteps >= 1800000:
            torch.save(metrics, os.path.join(save_dir, 'metrics_{}.pth'.format(total_numsteps // 2000000)))

        # writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {} | Avg. Reward: {} | Deploy: {} | Force: {}".format(episodes, round(avg_reward, 2), agent.num_deploy, agent.trigger_force_deploy))
        if args.record_feature_sim and 'feature sim' in metrics and len(metrics['feature sim']) >= 10:
            print("Feature Sim: {:.4f} | Reject: {}".format(np.mean(metrics['feature sim'][-10:]), agent.feature_reject))
            print("KL:", np.round(metrics['feature sim'][-5:], 4))
        if args.record_kl and 'kl' in metrics and len(metrics['kl']) >= 10:
            print("KL: {:.6f} | Reject: {}".format(np.mean(metrics['kl'][-10:]), agent.kl_reject))
            print("KL:", np.round(metrics['kl'][-5:], 6))
        if args.switching == 'kl' and len(metrics['kl when reset']) >= 5:
            print("Reset KL:", np.round(metrics['kl when reset'][-5:], 6))
        print("----------------------------------------")

env.close()
test_env.close()
