import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import seaborn as sns

prefix_dir = os.path.join(os.getcwd(), 'results')
env_list = ['Hopper', 'HalfCheetah', 'Walker2d', 'Ant', 'Swimmer']
colors = {'fix_1000': 'C0',
          'none': 'C1',
          'f90': 'C2',
          'fix_10000': 'C3',
          'adapt': 'C4'
         }
switching_list = colors.keys()
num_seed = 3
L = 150

N = len(env_list)
fig, axes = plt.subplots(4, (N - 1) // 2 + 1, figsize=(4 * N, 16))

fail_log = []

for n, env in enumerate(env_list):
    ax0, ax1 = axes[0 + 2 * (n % 2), n // 2], axes[1 + 2 * (n % 2), n // 2]
    for switching in switching_list:
        print(env, switching)
        try:
            sum_score, sum_deploy = np.zeros(150), np.zeros(150)
            list_score = []
            for i in range(L):
                list_score.append([])
            num_count = np.zeros(150).astype(np.long)
            real_num_seed = num_seed
            if switching in ['f87']:
                real_num_seed = 1
            for seed in range(real_num_seed):
                print(os.path.join(prefix_dir, env + '_scripts', switching + '.sh_seed_' + str(seed), 'metrics.pth'))
                m = torch.load(os.path.join(prefix_dir, env + '_scripts', switching + '.sh_seed_' + str(seed), 'metrics.pth'))
                for it, score, dep in zip(m['iter'], m['score'], m['deploy']):
                    k0 = int(it / 10000)
                    for k in range(k0 - 10, k0 + 11):
                        if k < 0 or k >= L: continue
                        list_score[k].append(score)
                        sum_score[k] += score
                        sum_deploy[k] += dep
                        num_count[k] += 1
            
            empty_place = num_count == 0
            slice = num_count > 0
            num_count[empty_place] = 1
                        
            real_T = 10000 * np.arange(L)
            avg_score = sum_score / num_count
            avg_deploy = sum_deploy / num_count
            avg_log_deploy = np.log(avg_deploy + 1) / np.log(10)
            std_score = np.zeros(L)
            
            for i in range(L):
                if len(list_score[i]) > 0:
                    std_score[i] = np.std(list_score[i]) 
            ax0.plot(real_T[slice], avg_score[slice], color=colors[switching], label=switching)
            ax0.fill_between(real_T[slice], avg_score[slice] - 0.2 * std_score[slice], avg_score[slice] + 0.2 * std_score[slice], color=colors[switching], alpha=0.15)
            ax1.plot(real_T[slice], avg_log_deploy[slice], color=colors[switching])
            
            ax0.legend()
            ax0.set_title(env)
            ax0.set_ylabel('test score')
            ax1.set_ylabel('log_10 number of deployments')
            ax1.set_xlabel('number of interactions')
            plt.title(env)
            print()

        except:
            fail_info = 'Fail in env {} with switching policy {}'.format(env, switching)
            print(fail_info)
            fail_log.append(fail_info)
            print()

plt.tight_layout()
plt.savefig('all_consise.png')
plt.close()

print(fail_log)