'''
Day 57

setting environmet for MDP model 
this env.py assumes that we know the boundaries
cf. dpn_env.py will be under the assumption that we do not know the boundaries

Learning source: https://github.com/marofmar/reinforcement_learning
forked from: SungchulLee/reinforcement_learning

Based on the Andrew Ng RL MLP map 

I studied by adding comments on the existing codes, so that I can review the lecture material of the professor Sungchul Lee (I took this course last semester)
'''

import numpy as np


class ENVIRONMENT:
    def __init__(self, policy=None, battery_consumption=-0.02, gamma=0.99): 	#set battery consupmsion!
        " Policy "
        self.policy = policy 
        
        " States "
        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.num_states = len(self.states)
        
        # "state 3" : (terminal-state) +1 reward, 	the goal and target spot to land in 
        # "state 6" : (terminal-state) -1 reward, 	the trap spot that we want to avoid 
        self.terminal_states = [3, 6] 	# whether we land on the target or trap, the game ends 
        self.win_state = self.terminal_states[0] # target 3
        self.lose_state = self.terminal_states[1] # trap 6
        
        self.non_terminal_states = [0, 1, 2, 4, 5, 7, 8, 9, 10] # the rest, which is safe to pass by 
        
        " Actions "
        self.actions = [0, 1, 2, 3]  # left, right, up, down
        self.num_actions = len(self.actions) # four different actions 

        " Transition Probabilities "
        self.P = np.empty((self.num_states, self.num_actions, self.num_states))

        #                     0   1   2   3   4   5   6   7   8   9  10
        self.P[ 0, 0, :] = [ .9,  0,  0,  0, .1,  0,  0,  0,  0,  0,  0] # I am in 0, and going left, then?
        self.P[ 0, 1, :] = [ .1, .8,  0,  0, .1,  0,  0,  0,  0,  0,  0]
        self.P[ 0, 2, :] = [ .9, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        self.P[ 0, 3, :] = [ .1, .1,  0,  0, .8,  0,  0,  0,  0,  0,  0]

        #                     0   1   2   3   4   5   6   7   8   9  10
        self.P[ 1, 0, :] = [ .8, .2,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        self.P[ 1, 1, :] = [  0, .2, .8,  0,  0,  0,  0,  0,  0,  0,  0]
        self.P[ 1, 2, :] = [ .1, .8, .1,  0,  0,  0,  0,  0,  0,  0,  0]
        self.P[ 1, 3, :] = [ .1, .8, .1,  0,  0,  0,  0,  0,  0,  0,  0] # I am in 1, and going down, the probability that I land on 0 is .1 and the prob for landing 1 is .8

        #                     0   1   2   3   4   5   6   7   8   9  10
        self.P[ 2, 0, :] = [  0, .8, .1,  0,  0, .1,  0,  0,  0,  0,  0]
        self.P[ 2, 1, :] = [  0,  0, .1, .8,  0, .1,  0,  0,  0,  0,  0]
        self.P[ 2, 2, :] = [  0, .1, .8, .1,  0,  0,  0,  0,  0,  0,  0]
        self.P[ 2, 3, :] = [  0, .1,  0, .1,  0, .8,  0,  0,  0,  0,  0]

        #                     0   1   2   3   4   5   6   7   8   9  10
        self.P[ 3, 0, :] = [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0]
        self.P[ 3, 1, :] = [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0]
        self.P[ 3, 2, :] = [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0]
        self.P[ 3, 3, :] = [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0]

        #                     0   1   2   3   4   5   6   7   8   9  10
        self.P[ 4, 0, :] = [ .1,  0,  0,  0, .8,  0,  0, .1,  0,  0,  0]
        self.P[ 4, 1, :] = [ .1,  0,  0,  0, .8,  0,  0, .1,  0,  0,  0]
        self.P[ 4, 2, :] = [ .8,  0,  0,  0, .2,  0,  0,  0,  0,  0,  0]
        self.P[ 4, 3, :] = [  0,  0,  0,  0, .2,  0,  0, .8,  0,  0,  0]

        #                     0   1   2   3   4   5   6   7   8   9  10
        self.P[ 5, 0, :] = [  0,  0, .1,  0,  0, .8,  0,  0,  0, .1,  0]
        self.P[ 5, 1, :] = [  0,  0, .1,  0,  0,  0, .8,  0,  0, .1,  0]
        self.P[ 5, 2, :] = [  0,  0, .8,  0,  0, .1, .1,  0,  0,  0,  0]
        self.P[ 5, 3, :] = [  0,  0,  0,  0,  0, .1, .1,  0,  0, .8,  0]

        #                     0   1   2   3   4   5   6   7   8   9  10
        self.P[ 6, 0, :] = [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0]
        self.P[ 6, 1, :] = [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0]
        self.P[ 6, 2, :] = [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0]
        self.P[ 6, 3, :] = [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0]

        #                     0   1   2   3   4   5   6   7   8   9  10
        self.P[ 7, 0, :] = [  0,  0,  0,  0, .1,  0,  0, .9,  0,  0,  0]
        self.P[ 7, 1, :] = [  0,  0,  0,  0, .1,  0,  0, .1, .8,  0,  0]
        self.P[ 7, 2, :] = [  0,  0,  0,  0, .8,  0,  0, .1, .1,  0,  0]
        self.P[ 7, 3, :] = [  0,  0,  0,  0,  0,  0,  0, .9, .1,  0,  0]

        #                     0   1   2   3   4   5   6   7   8   9  10
        self.P[ 8, 0, :] = [  0,  0,  0,  0,  0,  0,  0, .8, .2,  0,  0]
        self.P[ 8, 1, :] = [  0,  0,  0,  0,  0,  0,  0,  0, .2, .8,  0]
        self.P[ 8, 2, :] = [  0,  0,  0,  0,  0,  0,  0, .1, .8, .1,  0]
        self.P[ 8, 3, :] = [  0,  0,  0,  0,  0,  0,  0, .1, .8, .1,  0]

        #                     0   1   2   3   4   5   6   7   8   9  10
        self.P[ 9, 0, :] = [  0,  0,  0,  0,  0, .1,  0,  0, .8, .1,  0]
        self.P[ 9, 1, :] = [  0,  0,  0,  0,  0, .1,  0,  0,  0, .1, .8]
        self.P[ 9, 2, :] = [  0,  0,  0,  0,  0, .8,  0,  0, .1,  0, .1]
        self.P[ 9, 3, :] = [  0,  0,  0,  0,  0,  0,  0,  0, .1, .8, .1]

        #                     0   1   2   3   4   5   6   7   8   9  10
        self.P[10, 0, :] = [  0,  0,  0,  0,  0,  0, .1,  0,  0, .8, .1]
        self.P[10, 1, :] = [  0,  0,  0,  0,  0,  0, .1,  0,  0,  0, .9]
        self.P[10, 2, :] = [  0,  0,  0,  0,  0,  0, .8,  0,  0, .1, .1]
        self.P[10, 3, :] = [  0,  0,  0,  0,  0,  0,  0,  0,  0, .1, .9] # manual thorough detailed process done by a human to give a sacred valuable life to an agent
        
        " Reward "
        self.battery_consumption = float(battery_consumption) #battery consupmtion is also considered, so that we encourage the agent to get on the target asap
        self.reward = self.battery_consumption # pay for battery consumption # "Nothing is free." is the first lesson that the agent learn from this beautiful world.

        " Discount Factor "
        self.gamma = float(gamma) 

        " Current State "
        self.current_state = None # indicate initializaion by reset method is needed
        
        " Game Progress Indicator "
        self.done = None # indicate initializaion by reset method is needed

    def reset(self):
        """
        1. Choose an initial_state, or self.current_state randomly 
        from self.non_terminal_states.
        Since terminal-states are not in self.non_terminal_states,
        they are not chosen and cannot be an initial state.
        2. Set self.done as False.
        
        3. Set self.final_reward as None
        4. Return initial_state
        """
        self.current_state = np.random.choice(self.non_terminal_states) # randomly start from somewhere except 3 and 6 
        self.done = False
        self.final_reward = None
        return self.current_state, self.done
    
    def action(self):
        """
        Choose an action using self.policy 
        """
        action = np.random.choice(self.actions, p=self.policy[self.current_state,:]) # use the prob descriptions above 
        prob = self.policy[self.current_state, action] 
        return action, prob

    def step(self, action, prob=None):
        # check self.current_state is in self.non_terminal_states
        # otherwise raise error
        if self.current_state is None:
            raise ValueError("Current state should be initialized by reset method")
        if self.current_state in self.terminal_states:
            raise ValueError("Current state is terminal state")

        action = int(action) # make sure action is an integer
        self.info = "prob " + str(prob)
        
        next_state = np.random.choice(self.num_states,
                                      p=self.P[self.current_state, action, :])
        
        if next_state == self.win_state:
            #self.reward += 1.       # happy ending
            self.final_reward = 1.   # happy ending
            self.done = True         # game over
        if next_state == self.lose_state:
            #self.reward += -1.       # unhappy ending
            self.final_reward = -1.   # unhappy ending
            self.done = True          # game over

        return self.reward, next_state, self.done, self.info, self.final_reward

    def random_action(self):
        """
        Return randomly chosen action from [0,1,2,3] or left, right, up, down
        """
        return np.random.choice(self.num_actions)


