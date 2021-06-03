import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import os
import numpy as np
import seaborn as sns

prefix_dir = os.path.join(os.getcwd(), 'results')
#env_list = ['Hopper', 'HalfCheetah', 'Walker2d', 'Ant', 'Swimmer']
env_list = ['Swimmer', 'HalfCheetah', 'Ant']#, 'Walker2d', 'Hopper']
all_switching = ['none', 'f90', 'kl150', 'det', 'visit_2', 'fix_1000', 'fix_10000', 'adapt']
switching_list = ['none', 'f90']#, 'kl150', 'det', 'visit_2', 'adapt']

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
fig = plt.figure(figsize=(12, 6))
axes = fig.add_gridspec(ncols=3, nrows=2, height_ratios=[1, 8], hspace=0.1)

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
            sum_score, sum_deploy = np.zeros(150), np.zeros(150)
            list_score = []
            for i in range(L):
                list_score.append([])
            list_deploy = []
            for i in range(L):
                list_deploy.append([])
            num_count = np.zeros(150).astype(np.long)
            real_num_seed = num_seed
            for seed in range(real_num_seed):
                print(os.path.join(prefix_dir, env + '_scripts', switching + '.sh_seed_' + str(seed), 'metrics.pth'))
                m = torch.load(os.path.join(prefix_dir, env + '_scripts', switching + '.sh_seed_' + str(seed), 'metrics.pth'))
                for it, score, dep in zip(m['iter'], m['score'], m['deploy']):
                    k0 = int(it / 10000)
                    for k in range(k0 - 10, k0 + 11):
                        if k < 0 or k >= L: continue
                        list_score[k].append(score)
                        list_deploy[k].append(dep)
                        sum_score[k] += score
                        sum_deploy[k] += dep
                        num_count[k] += 1
            
            empty_place = num_count == 0
            slice = num_count > 0
            num_count[empty_place] = 1
                        
            real_T = 10000 * np.arange(L)
            avg_score = sum_score / num_count
            avg_deploy = sum_deploy / num_count
            std_score = np.zeros(L)
            std_deploy = np.zeros(L)
            
            for i in range(L):
                if len(list_score[i]) > 0:
                    std_score[i] = np.std(list_score[i]) 
            for i in range(L):
                if len(list_deploy[i]) > 0:
                    std_deploy[i] = np.std(list_deploy[i]) 
            ax0.plot(real_T[slice], avg_score[slice], color=colors[switching])
            ax0.fill_between(real_T[slice], avg_score[slice] - std_score[slice], avg_score[slice] + std_score[slice], color=colors[switching], alpha=0.1)
            ax1.plot(real_T[slice], avg_deploy[slice], color=colors[switching])
            ax1.fill_between(real_T[slice], avg_deploy[slice] - std_deploy[slice], avg_deploy[slice] + std_deploy[slice], color=colors[switching], alpha=0.1)
            print()

        except:
            fail_info = 'Fail in env {} with switching policy {}'.format(env, switching)
            print(fail_info)
            fail_log.append(fail_info)
            print()
    ax0.set_title(env)
    if n % 3 == 0:
        ax0.set_ylabel('Reward score')
        ax1.set_ylabel('Switching\n Cost')
    if N // 3 == n // 3:
        ax1.set_xlabel('Step')

fig.savefig('mujoco.png', bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.close()

print(fail_log)