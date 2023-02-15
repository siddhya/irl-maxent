"""
IPD-World Markov Decision Processes (MDPs).

The MDPs in this module are actually not complete MDPs, but rather the
sub-part of an MDP containing states, actions, and transitions (including
their probabilistic character). Reward-function and terminal-states are
supplied separately.

"""

import numpy as np
from itertools import product


class IPDWorld:
    """
    Iterated Prisoner's Dilemma world MDP.

    Args:
        memory: how many time steps do you remember.

    Attributes:
        n_states: The number of states of this MDP.
        n_actions: The number of actions of this MDP.
        p_transition: The transition probabilities as table. The entry
            `p_transition[from, to, a]` contains the probability of
            transitioning from state `from` to state `to` via action `a`.
        size: The width and height of the world.
        actions: The actions of this world as paris, indicating the
            direction in terms of coordinates.
    """

    def __init__(self, memory = 1):
        self.memory = memory
        self.actions = ['C', 'D']
        self.states = []
        
        # initial state
        # FIXME: Currently opponant starts with cooperate.
        # We can also make it so that agent starts.
        self.states.append({"opp_last_act": 0,
                            "agt_last_act": 0,
                            "opp_curr_act": 'C'})
        # states
        for opp_last_act in self.actions:
            for agt_last_act in self.actions:
                for opp_curr_act in self.actions:
                    self.states.append({"opp_last_act": opp_last_act,
                                        "agt_last_act": agt_last_act,
                                        "opp_curr_act": opp_curr_act})

        self.n_states = len(self.states)
        self.n_actions = len(self.actions)

        self.p_transition = self._transition_prob_table()

    def _transition_prob_table(self):
        """
        Builds the internal probability transition table.

        Returns:
            The probability transition table of the form

                [state_from, state_to, action]

            containing all transition probabilities. The individual
            transition probabilities are defined by `self._transition_prob'.
        """
        table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))

        s1, s2, a = range(self.n_states), range(self.n_states), range(self.n_actions)
        for s_from, s_to, a in product(s1, s2, a):
            table[s_from, s_to, a] = self._transition_prob(s_from, s_to, a)

        return table

    def _is_legal(self, s_from, a, s_to):
        if (s_to['opp_last_act'] != s_from['opp_curr_act'] or
            s_to['agt_last_act'] != a):
            return False
        return True

    def _index_to_values(self, s_from_idx, a_idx, s_to_idx = None):
        if s_to_idx is not None:
            return self.states[s_from_idx], self.actions[a_idx], self.states[s_to_idx]
        else:
            return self.states[s_from_idx], self.actions[a_idx]

    def _values_to_index(self, s):
        return next((idx for (idx, d) in enumerate(self.states) if d == s))

    def state_features(self):
        """
        Return the feature matrix assigning each state with an individual
        feature (i.e. initial_state, opp_last_act, agt_last_act, opp_curr_act).

        Rows represent individual states, columns the feature entries.

        Args:
            world: An IPDWorld instance for which the feature-matrix should be
                computed.

        Returns:
            The coordinate-feature-matrix for the specified world.
        """
        # 2 features each for every memory slot and 1 feature for initial state
        f = np.zeros(shape=(self.n_states, 1 + (self.memory * 2 * 2) + 2))
        for i, s in enumerate(self.states):
            if s["opp_last_act"] == 0:
                #initial state
                f[i][0] = 1
                if s["opp_curr_act"] == 'C':
                    f[i][5] = 1
                else:
                    f[i][6] = 1
            else:
                if s["opp_last_act"] == 'C':
                    f[i][1] = 1
                else:
                    f[i][2] = 1
                if s["agt_last_act"] == 'C':
                    f[i][3] = 1
                else:
                    f[i][4] = 1
                if s["opp_curr_act"] == 'C':
                    f[i][5] = 1
                else:
                    f[i][6] = 1
        return f

    def __repr__(self):
        return "IPDWorld(memory={})".format(self.memory)

class RandomIPDWorld(IPDWorld):
    """
    The opponent is a random player who chooses cooprate
    or defect with same probability.
    """
    def _transition_prob(self, s_from_idx, s_to_idx, a_idx):
        """
        Compute the transition probability for a single transition.

        Args:
            s_from: The state in which the transition originates.
            s_to: The target-state of the transition.
            a: The action via which the target state should be reached.

        Returns:
            The transition probability from `s_from` to `s_to` when taking
            action `a`.
        """
        s_from, a, s_to = self._index_to_values(s_from_idx, a_idx, s_to_idx)
        if not self._is_legal(s_from, a, s_to):
            return 0.0
        # Use a random policy for now so half probabilty of choosing either of the two actions
        return 0.5
    
    def state_index_transition(self, s_idx, a_idx):
        """
        Perform action `a` at state `s` and return the intended next state.

        Does not take into account the transition probabilities. Instead it
        just returns the intended outcome of the given action taken at the
        given state, i.e. the outcome in case the action succeeds.

        Args:
            s: The state at which the action should be taken.
            a: The action that should be taken.

        Returns:
            The next state as implied by the given action and state.
        """
        s, a = self._index_to_values(s_idx, a_idx)
        s_next = {"opp_last_act": s["opp_curr_act"],
                  "agt_last_act": a,
                  # Opponent is playing uniform random policy
                  "opp_curr_act": np.random.choice(self.actions)}
        return self._values_to_index(s_next)

class TitTatIPDWorld(IPDWorld):
    """
    The opponent is a tit-for-tat player who chooses cooprate
    or defect if agent chooses cooerate or defect respectively.
    """
    def _transition_prob(self, s_from_idx, s_to_idx, a_idx):
        """
        Compute the transition probability for a single transition.

        Args:
            s_from: The state in which the transition originates.
            s_to: The target-state of the transition.
            a: The action via which the target state should be reached.

        Returns:
            The transition probability from `s_from` to `s_to` when taking
            action `a`.
        """
        s_from, a, s_to = self._index_to_values(s_from_idx, a_idx, s_to_idx)
        if not self._is_legal(s_from, a, s_to):
            return 0.0
        # tit-for-tat
        if s_to['opp_curr_act'] == a:
            return 1.0
        return 0.0

    def state_index_transition(self, s_idx, a_idx):
        """
        Perform action `a` at state `s` and return the intended next state.

        Does not take into account the transition probabilities. Instead it
        just returns the intended outcome of the given action taken at the
        given state, i.e. the outcome in case the action succeeds.

        Args:
            s: The state at which the action should be taken.
            a: The action that should be taken.

        Returns:
            The next state as implied by the given action and state.
        """
        s, a = self._index_to_values(s_idx, a_idx)
        s_next = {"opp_last_act": s["opp_curr_act"],
                  "agt_last_act": a,
                  # Opponent is playing tit-for-tat
                  "opp_curr_act": a}
        return self._values_to_index(s_next)