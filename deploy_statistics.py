import matplotlib.pyplot as plt
import torch
import os
import numpy as np

prefix_dir = os.path.join(os.getcwd(), 'results')

env_list = ['Hopper', 'HalfCheetah', 'Walker2d', 'Ant', 'Swimmer']
switching = 'f90'

for env in env_list:
    total_deploy = 0
    for seed in range(3):
        m = torch.load(os.path.join(prefix_dir, env + '_scripts', switching + '.sh_seed_' + str(seed), 'metrics.pth'))
        total_deploy += m['deploy'][-1]
    print('Env: {}, avg total deploy = {}'.format(env, total_deploy / 3))
