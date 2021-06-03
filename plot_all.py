import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy
import torch
import os
import numpy as np
import seaborn as sns

prefix_dir = os.path.join(os.getcwd(), 'results')
#env_list = ['Hopper', 'HalfCheetah', 'Walker2d', 'Ant', 'Swimmer']
env_list = ['Swimmer', 'HalfCheetah', 'Ant', 'Walker2d', 'Hopper']
all_switching = ['none', 'f90', 'kl150', 'det', 'visit_2', 'fix_1000', 'fix_10000', 'adapt']
switching_list = ['none', 'f90', 'kl150', 'det', 'visit_2', 'adapt']

sns_palette = sns.color_palette()
colors = {}
for n, switching in enumerate(switching_list):
    colors[switching] = sns_palette[n]

plot_name = {'none': 'None',
             'f90': 'Feature',
             'kl150': 'KL Divergence',
             'det': 'Info',
             'visit_2': 'Visitation',
             'fix_1000': 'FIX_1000',
             'fix_10000': 'FIX_10000',
             'adapt': 'Linear',
            }
 
num_seed = 3
L = 150

N = len(env_list)
fig = plt.figure(figsize=(12, 8))
axes = fig.add_gridspec(ncols=3, nrows=3, height_ratios=[1, 8, 8], hspace=0.3)

legend_ax = fig.add_subplot(axes[0, :])
for sp in legend_ax.spines.values():
    sp.set_visible(False)
legend_ax.set_xticks([])
legend_ax.set_yticks([])
patches = [mpatches.Patch(color=colors[switching], label=plot_name[switching]) for switching in switching_list]
legend_ax.legend(handles=patches, ncol = len(switching_list), mode="expand", edgecolor="white")
    
fail_log = []

for n, env in enumerate(env_list):
    env_ax = axes[n + 3].subgridspec(ncols=1, nrows=2, hspace=0.4)
    ax0, ax1 = fig.add_subplot(env_ax[0]), fig.add_subplot(env_ax[1])
    ax1.set_yscale('log')
    for switching in switching_list:
        print(env, switching)
        try:
            real_num_seed = num_seed
            real_T = np.linspace(0, 1.5e6, 200)
            real_reward = []
            real_deploy = []
            for seed in range(real_num_seed):
                print(os.path.join(prefix_dir, env + '_scripts', switching + '.sh_seed_' + str(seed), 'metrics.pth'))
                m = torch.load(os.path.join(prefix_dir, env + '_scripts', switching + '.sh_seed_' + str(seed), 'metrics.pth'))
                m_len = len(m['iter'])
                m_iter = np.zeros(m_len + 2)
                m_iter[1:-1] = np.array(m['iter'])
                m_iter[-1] = 1.5e6
                m_reward = np.zeros(m_len + 2)
                m_reward[1:-1] = np.array(m['score'])
                m_reward[-1] = m_reward[-2]
                m_deploy = np.zeros(m_len + 2)
                m_deploy[1:-1] = np.array(m['deploy'])
                m_deploy[-1] = m_deploy[-2]
                f_reward = scipy.interpolate.interp1d(m_iter, m_reward)
                real_reward.append(f_reward(real_T))
                f_deploy = scipy.interpolate.interp1d(m_iter, m_deploy)
                real_deploy.append(f_deploy(real_T))
            real_reward, real_deploy = np.stack(real_reward), np.stack(real_deploy)
            avg_reward, std_reward = real_reward.mean(0), real_reward.std(0)
            avg_deploy, std_deploy = real_deploy.mean(0), real_deploy.std(0)
                
            ax0.plot(real_T, avg_reward, color=colors[switching])
            ax0.fill_between(real_T, avg_reward - std_reward, avg_reward + std_reward, color=colors[switching], alpha=0.1)
            ax1.plot(real_T, avg_deploy, color=colors[switching])
            ax1.fill_between(real_T, avg_deploy - std_deploy, avg_deploy + std_deploy, color=colors[switching], alpha=0.1)
            print()

        except:
            fail_info = 'Fail in env {} with switching policy {}'.format(env, switching)
            print(fail_info)
            fail_log.append(fail_info)
            print()
    ax0.set_title(env)
    if n % 3 == 0:
        ax0.set_ylabel('Reward')
        ax1.set_ylabel('Switching\n Cost')
    if N // 3 == n // 3:
        ax1.set_xlabel('Step')

fig.savefig('mujoco_all.png', bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.close()

print(fail_log)