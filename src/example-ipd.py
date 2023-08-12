#!/usr/bin/env python

from irl_maxent import ipdworld as W
from irl_maxent import maxent as M
from irl_maxent import plot as P
from irl_maxent import trajectory as T
from irl_maxent import solver as S
from irl_maxent import optimizer as O

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def setup_mdp():
    """
    Set-up our MDP/IPDWorld
    """
    # create our world
    world = W.TitTatIPDWorld()

    # set up the reward function
    reward = np.zeros(world.n_states)
    for i in range(world.n_states):
        if world.states[i]['agt_last_act'] == 'C' and world.states[i]['opp_curr_act'] == 'C':
            reward[i] = 5
        elif world.states[i]['agt_last_act'] == 'D' and world.states[i]['opp_curr_act'] == 'C':
            reward[i] = 3
        elif world.states[i]['agt_last_act'] == 'C' and world.states[i]['opp_curr_act'] == 'D':
            reward[i] = 0
        elif world.states[i]['agt_last_act'] == 'D' and world.states[i]['opp_curr_act'] == 'D':
            reward[i] = 1

    # Infinite horizon so set the number of iterations to run
    terminal = 10
    # Terminate when all three memory places are cooperate
    terminal = []
    for i, s in enumerate(world.states):
        if s["opp_last_act"] == 'C' and s["agt_last_act"] == 'C' and s["opp_curr_act"] == 'C':
            terminal.append(i)

    return world, reward, terminal


def generate_trajectories(world, reward, terminal):
    """
    Generate some "expert" trajectories.
    """
    # parameters
    n_trajectories = 200
    discount = 0.7
    weighting = lambda x: x**5

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    initial[0] = 1

    # generate trajectories
    value = S.value_iteration(world.p_transition, reward, discount)
    policy = S.stochastic_policy_from_value(world, value, w=weighting)
    policy_exec = T.stochastic_policy_adapter(policy)
    tjs = list(T.generate_trajectories(n_trajectories, world, policy_exec, initial, terminal))

    return tjs, policy


def maxent(world, terminal, trajectories):
    """
    Maximum Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = world.state_features()

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = M.irl(world.p_transition, features, terminal, trajectories, optim, init)

    return reward


def maxent_causal(world, terminal, trajectories, discount=0.7):
    """
    Maximum Causal Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = world.state_features()

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = M.irl_causal(world.p_transition, features, terminal, trajectories, optim, init, discount)

    return reward


def main():
    # common style arguments for plotting
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    # set-up mdp
    world, reward, terminal = setup_mdp()

    # show our original reward
    #ax = plt.figure(num='Original Reward').add_subplot(111)
    #P.plot_state_values(ax, world, reward, **style)
    #plt.draw()

    # generate "expert" trajectories
    trajectories, expert_policy = generate_trajectories(world, reward, terminal)

    # show our expert policies
    #ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
    #P.plot_stochastic_policy(ax, world, expert_policy, **style)

    #for t in trajectories:
    #    P.plot_trajectory(ax, world, t, lw=5, color='white', alpha=0.025)

    #plt.draw()

    # maximum entropy reinforcement learning (non-causal)
    reward_maxent = maxent(world, terminal, trajectories)

    # show the computed reward
    #ax = plt.figure(num='MaxEnt Reward').add_subplot(111)
    #P.plot_state_values(ax, world, reward_maxent, **style)
    #plt.draw()

    # maximum casal entropy reinforcement learning (non-causal)
    reward_maxcausal = maxent_causal(world, terminal, trajectories)

    # show the computed reward
    #ax = plt.figure(num='MaxEnt Reward (Causal)').add_subplot(111)
    #P.plot_state_values(ax, world, reward_maxcausal, **style)
    #plt.draw()

    #plt.show()
    output = pd.DataFrame()
    output = output.append(world.states)
    output['orig_reward'] = reward
    output['maxent_reward'] = reward_maxent
    output['maxcausal_reward'] = reward_maxcausal
    print(output)

    fig = plt.figure()
    fig.suptitle('Interated Prisoners Dilemma (Tit-for-Tat) using IRL')
    P.plot_rewards(fig, reward, reward_maxent, reward_maxcausal)
    plt.show()

if __name__ == '__main__':
    main()
