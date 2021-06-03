import numpy as np
import torch
import os

class HashTable(object):
  def __init__(self, args, state_dim, act_dim):
    self.table = {}
    self.state_action_table = {}
    self.state_action_count = 0
    self.hash_dim = args.hash_dim
    self.state_dim = state_dim
    self.act_dim = act_dim
    self.A = np.random.randn(self.hash_dim, self.state_dim)
    self.B = np.random.randn(self.hash_dim, self.state_dim + self.act_dim)
    self.info_matrix = {}
    self.previous_info_value = {}
    self.previous_info_det = {}
    self.cur_info_index = None      

  def step(self, state, action, count_action_state=True, need_info_matrix=False, info_index=0):
    hash_item, state_action_hash_item, hidden_with_action = self.hash_state(state, action)
    if hash_item in self.table:
      self.table[hash_item] += 1
    else:
      self.table[hash_item] = 1
    
    if need_info_matrix:
      self.step_info_matrix(hidden_with_action, info_index)   
    
    if count_action_state:
      if state_action_hash_item in self.state_action_table:
        self.state_action_table[state_action_hash_item] += 1
      else:
        self.state_action_table[state_action_hash_item] = 1
      self.state_action_count = self.state_action_table[state_action_hash_item]
    return self.table[hash_item]

  def hash_state(self, state, action):
    state_hidden = (self.A * state).sum(-1)
    state_hash = ''.join(map(lambda x: 'a' if x > 0 else 'b', state_hidden))
    state_action_hidden = (self.B * np.concatenate([state, action], 0)).sum(-1)
    state_action_hash = ''.join(map(lambda x: 'a' if x > 0 else 'b', state_action_hidden))
    return state_hash, state_action_hash, torch.from_numpy(state_action_hidden).contiguous().view(-1, 1)
  
  def step_info_matrix(self, hidden_with_action, info_index):
    cur_matrix = hidden_with_action @ hidden_with_action.T
    if self.info_matrix.get(info_index, None) is None:
      self.info_matrix[info_index] = cur_matrix
    else:
      self.info_matrix[info_index] += cur_matrix
    self.cur_info_index = info_index
  
  @property
  def info_matrix_value(self):
    (evals, _) = torch.eig(self.info_matrix[self.cur_info_index], eigenvectors=False)
    evals = evals[:, 0]
    cur_info_value = evals.abs().min().item()
    previous_value = self.previous_info_value.get(self.cur_info_index, None)
    out_flag = previous_value is None or previous_value == 0 or cur_info_value / previous_value >= 2
    if out_flag:
      self.previous_info_value[self.cur_info_index] = cur_info_value
    return cur_info_value, out_flag
  
  @property
  def info_matrix_det(self):
    A = self.info_matrix[self.cur_info_index]
    n = A.size(0)
    I_n = torch.from_numpy(np.identity(n)).to(A.dtype)
    cur_info_det = torch.det(A + I_n).abs()
    previous_det = self.previous_info_det.get(self.cur_info_index, None)
    out_flag = previous_det is None or previous_det == 0 or cur_info_det / previous_det >= 2
    if out_flag:
      self.previous_info_det[self.cur_info_index] = cur_info_det
    return cur_info_det, out_flag

  def save(self, path, name='hash.pth'):
    torch.save({'A': self.A, 'B': self.B, 'table': self.table, 'state_action_table': self.state_action_table}, os.path.join(path, name))

  def load(self, path, name='hash.pth'):
    state = torch.load(os.path.join(path, name), map_location='cpu')
    self.A = state['A']
    self.B = state['B']
    self.table = state['table']
    self.state_action_table = state.get('state_action_table', {})

