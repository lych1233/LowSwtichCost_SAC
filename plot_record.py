import matplotlib.pyplot as plt
import torch
import os
import numpy as np

prefix_dir = os.path.join(os.getcwd(), 'results')

env_list = ['Hopper', 'HalfCheetah', 'Walker2d']
ver = {'Hopper': 4,
        'HalfCheetah': 4,
        'Walker2d': 4
      }
switching = 'record'
seed = 0

N = 3
fig, axes = plt.subplots(3, N, figsize=(6 * N, 8))

for n, env in enumerate(env_list):
    ax0, ax1, ax2 = axes[0, n], axes[1, n], axes[2, n]
    m = torch.load(os.path.join(prefix_dir, env + '_scripts', switching + '.sh_seed_' + str(seed), 'metrics_{}.pth'.format(ver[env])))
    print(len(m['iter']), len(m['feature try T']), len(m['kl try T']))
    ax0.plot(m['iter'], m['score'])
    ax1.plot(m['feature try T'][::10000], m['feature sim'][::10000])
    ax2.plot(m['kl try T'][::10000], m['kl'][::10000])
    if n == 0:
        ax0.set_ylabel('score')
        ax1.set_ylabel('feature sim')
        ax2.set_ylabel('kl')
plt.savefig('long.png')    