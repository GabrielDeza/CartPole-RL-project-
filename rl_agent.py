import random
import gym
import math
import numpy as np

"""
Agent Description:
I used a Q-Learning approach where I constructed a table to match all possible
states to actions, and update every entry using the Q Learning formula/ bellman
equation that I learned in class as well as on Q Learning Wikipedia page. 
Since the state values are continous, I divided the state space into bins, 
as described in the original Barto and Sutton Paper [1]. This allowed me to 
make a multidimensional table for all possible {State, Action} pairs. I used 
an adaptive learning rate and exploration rate, such that picking a random 
action to explore the state space is less likely as the number of episodes 
progresses. The learning rate changes similiarly, it decreases as number of
episodes increases. Furthermore, due to the winstreak condition, my agent would
sometimes pick a random (bad) move, which would break my winstreak condition. 
To overcome this, I stop randomly exploring if I have a winstreak greater 
than 25, and if I have an average reward less then 265 over ~300 episodes,
I restart the whole learning process (as my model sometimes gets stuck.
Lastly, I did a parameter search for all parameters to get 15/15 
somewhat reliably.
[1]: A. G. Barto, R. S. Sutton, and C. W. Anderson, “Neuronlike adaptive 
elements that can solve difficult learning control problems,”
IEEE Transactions on Systems, Man, and Cybernetics, 
vol. SMC-13, pp. 834–846, Sept./Oct. 1983.
"""

class CartPoleAgent:

    def __init__(self, observation_space, action_space):
        #----- TODO: Add your code here. -----
        # Store observation space and action space.
        self.observation_space = observation_space
        self.action_space = action_space

        #Discretization and Q-Table initialization
        self.bins = (0,0,5,11,self.action_space.n)
        self.q_table = np.zeros((1,1,6,12,self.action_space.n))

        #Hyper Parameter
        self.min_alpha = 0.1 #learning rate
        self.min_epsilon = 0.01 #exploration rate
        self.adaptive_eps = 24 #adaptive parameter for eps
        self.adaptive_alpha = 23 #adaptive parameter for learning rate

        self.reset_calls = 0
        self.action_calls = 0

        self.stop = False
        self.eps_reward = 0
        self.good_eps_in_a_row = 0
        self.rew = []

    def action(self, state):
        """Choose an action from set of possible actions."""
        #----- TODO: Add your code here. -----
        self.current_state = self.discretize(state)
        #randomly expore
        if np.random.random() <= self.epsilon and self.stop == False:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[self.current_state])

    def reset(self):
        """Reset the agent, if desired."""
        #----- TODO: Add your code here. -----
        self.epsilon, self.alpha = self.adaptive_parameter()
        if self.reset_calls != 0:
            self.rew.append(self.eps_reward)
        self.reset_calls += 1
        if self.eps_reward > 265:
            self.good_eps_in_a_row += 1
        else:
            self.good_eps_in_a_row = 0
            self.stop = False
        if self.good_eps_in_a_row > 50:
            self.stop = True
        if self.reset_calls > 350 and np.average(self.rew[-300:]) < 300:
            self.reset_calls = 0
        self.eps_reward = 0

    def update(self, state, action, reward, state_next, terminal):
        """Update the agent internally after an action is taken."""
        #----- TODO: Add your code here. -----
        self.new_state = self.discretize(state_next)
        self.update_q(self.current_state,action,reward,self.new_state)
        if reward == 1:
            self.eps_reward += 1

    #--------- Added functions -----------------
    def discretize(self, obs):
        #since velocity and angular velocity are basically unbounded (infinity), before we disceretize,
        #I put a limit on, since when these values are extreme, the model probably already terminated.
        #also since it is symetric, dont need to define lower bound

        x_upper_bound = self.observation_space.high[0]
        v_upper_bound = 1.5
        theta_upper_bound = self.observation_space.high[2]
        w_upper_bound =(np.pi *(1/3))
        new_x = min(self.bins[0], \
                    max(0,int(round(self.bins[0] *((obs[0] + x_upper_bound) / (2*x_upper_bound))))))
        new_v = min(self.bins[1], \
                    max(0, int(round(self.bins[1] * ((obs[1] + v_upper_bound) / (2 * v_upper_bound))))))
        new_theta = min(self.bins[2], \
                        max(0, int(round(self.bins[2] * ((obs[2] + theta_upper_bound) / (2 * theta_upper_bound))))))
        new_w = min(self.bins[3], \
                    max(0, int(round(self.bins[3] * ((obs[3] + w_upper_bound) / (2 * w_upper_bound))))))
        #print(tuple([new_x, new_v,new_theta, new_w]))
        return tuple([new_x, new_v,new_theta, new_w])

    def update_q(self, state_old, action, reward, state_new):
        #standard Q learning update:
        # new q value is old value plus reward and greatest future rewards possible from that new state,
        # all weighted by an adaptive learning rate
        self.q_table[state_old][action] = (1-self.alpha)*(self.q_table[state_old][action]) \
                                    + self.alpha * (reward + np.max(self.q_table[state_new]))
    def adaptive_parameter(self):
        epoch = self.reset_calls + 1
        adaptive_e = 1.0 - np.log10((epoch) / self.adaptive_eps)
        adaptive_lr = 1.0 - np.log10((epoch) / self.adaptive_alpha)
        best_eps = max(self.min_epsilon, adaptive_e)
        best_lr =  max(self.min_alpha, adaptive_lr)
        return best_eps, best_lr

