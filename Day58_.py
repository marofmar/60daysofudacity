'''
Day 58
1. Grid world
2. Iterative policy evaluation
- Setting Grid world 
- source: https://github.com/lazyprogrammer/machine_learning_examples/blob/master/rl/grid_world.py
'''

'''
1. <<GRDI SETTING>>: Grid Wrold 
'''

# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np


class Grid: # Environment
  def __init__(self, rows, cols, start): # unlike the standard grid format, we can set up new ones
    self.rows = rows
    self.cols = cols
    self.i = start[0]
    self.j = start[1]

  def set(self, rewards, actions):
    # rewards should be a dict of: (i, j): r (row, col): reward
    # actions should be a dict of: (i, j): A (row, col): list of possible actions
    self.rewards = rewards
    self.actions = actions

  def set_state(self, s): # set the coordinates (i, j) 
    self.i = s[0]
    self.j = s[1]

  def current_state(self):
    return (self.i, self.j)

  def is_terminal(self, s):
    return s not in self.actions

  def move(self, action):
    # check if legal move first
    if action in self.actions[(self.i, self.j)]: # up and down change 'i', right and left change 'j'
      if action == 'U':
        self.i -= 1
      elif action == 'D':
        self.i += 1
      elif action == 'R':
        self.j += 1
      elif action == 'L':
        self.j -= 1
    # return a reward (if any)
    return self.rewards.get((self.i, self.j), 0)

  def undo_move(self, action):
    # these are the opposite of what U/D/L/R should normally do
    if action == 'U':
      self.i += 1
    elif action == 'D':
      self.i -= 1
    elif action == 'R':
      self.j -= 1
    elif action == 'L':
      self.j += 1
    # raise an exception if we arrive somewhere we shouldn't be
    # should never happen
    assert(self.current_state() in self.all_states())

  def game_over(self):
    # returns true if game is over, else false
    # true if we are in a state where no actions are possible
    return (self.i, self.j) not in self.actions # which are terminal points, 3(goal) and 6(trap) 

  def all_states(self):
    # possibly buggy but simple way to get all states
    # either a position that has possible next actions
    # or a position that yields a reward
    return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid():
  # define a grid that describes the reward for arriving at each state
  # and possible actions at each state
  # the grid looks like this
  # x means you can't go there
  # s means start position
  # number means reward at that state
  # .  .  .  1
  # .  x  . -1
  # s  .  .  .
  g = Grid(3, 4, (2, 0))
  rewards = {(0, 3): 1, (1, 3): -1} ## target: 3, which is (0,3) in coordinates format, trap: 6, in coordinates format(1, 3) 
  actions = {
    (0, 0): ('D', 'R'), ## possible action sets that each point can have
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (1, 0): ('U', 'D'),
    (1, 2): ('U', 'D', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U'),
  }
  g.set(rewards, actions)
  return g


def negative_grid(step_cost=-0.1):
  # in this game we want to try to minimize the number of moves
  # so we will penalize every move
  g = standard_grid()
  g.rewards.update({
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
  })
  return g


'''
==========================================================================
2. <<Iterative policy evaluation>>
'''
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from grid_world import standard_grid

SMALL_ENOUGH = 1e-3 # threshold for convergence

def print_values(V, g):
  for i in range(g.rows):
    print("---------------------------")
    for j in range(g.cols):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")


def print_policy(P, g):
  for i in range(g.rows):
    print("---------------------------")
    for j in range(g.cols):
      a = P.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")

if __name__ == '__main__':
  # iterative policy evaluation
  # given a policy, let's find it's value function V(s)
  # we will do this for both a uniform random policy and fixed policy
  # NOTE:
  # there are 2 sources of randomness
  # p(a|s) - deciding what action to take given the state
  # p(s',r|s,a) - the next state and reward given your action-state pair
  # we are only modeling p(a|s) = uniform
  # how would the code change if p(s',r|s,a) is not deterministic?
  grid = standard_grid()

  # states will be positions (i,j)
  # simpler than tic-tac-toe because we only have one "game piece"
  # that can only be at one position at a time
  states = grid.all_states()

  ### uniformly random actions ###
  # initialize V(s) = 0
  V = {}
  for s in states:
    V[s] = 0
  gamma = 1.0 # discount factor
  # repeat until convergence
  while True:
    biggest_change = 0
    for s in states:
      old_v = V[s]

      # V(s) only has value if it's not a terminal state
      if s in grid.actions:

        new_v = 0 # we will accumulate the answer
        p_a = 1.0 / len(grid.actions[s]) # each action has equal probability
        for a in grid.actions[s]:
          grid.set_state(s)
          r = grid.move(a)
          new_v += p_a * (r + gamma * V[grid.current_state()])
        V[s] = new_v
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))

    if biggest_change < SMALL_ENOUGH:
      break
  print("values for uniformly random actions:")
  print_values(V, grid)
  print("\n\n")

  ### fixed policy ###
  policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
  }
  print_policy(policy, grid)

  # initialize V(s) = 0
  V = {}
  for s in states:
    V[s] = 0

  # let's see how V(s) changes as we get further away from the reward
  gamma = 0.9 # discount factor

  # repeat until convergence
  while True:
    biggest_change = 0
    for s in states:
      old_v = V[s]

      # V(s) only has value if it's not a terminal state
      if s in policy:
        a = policy[s]
        grid.set_state(s)
        r = grid.move(a)
        V[s] = r + gamma * V[grid.current_state()]
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))

    if biggest_change < SMALL_ENOUGH:
      break
  print("values for fixed policy:")
  print_values(V, grid)