#!/usr/bin/env python

from python_speech_features import mfcc, delta, logfbank
import scipy.io.wavfile as wav
from scipy.cluster.vq import kmeans2, whiten, vq
import numpy as np

"""
Validation of the hidden markov model using the ice cream example.
Similar to the problem presented here: https://nadesnotes.wordpress.com/tag/hidden-markov-models/

Model:
n hiddens states
m observations

transition matrix: of size (n+2)x(n+2) where item at index (i,j) is the 
                   probability of transitioning from state i to state j
emissions matrix: of size mxn where item at index (i,j) is the probability 
                  of seeing ith observation in j state.
"""
# TODO: Create V, a bunch of observation codebook. It contains v_k

class HMM(object):

    def __init__(self, name=None, transitions=None, emissions=None, states=None):

        self.name = name
        self.transitions = transitions # Matrix of transision probabilities from one state to another. Size # of states by # of states
        self.emissions = emissions # Matrix of emission probabilities. Size # of states by # of observation symbol
        self.states = states # Includes start (q0) and end (qF) states. Along with N hidden states
        self.state_len = len(self.states)
        self.final_state_index = self.state_len-1

        # Gets set while training
        self.observations = None
        self.observation_len = 0
        self.final_observation_index = 0

        self.alpha = None
        self.beta = None
        self.zeta = None
        self.gamma = None

    def emission_prob(self, state, time): # TODO: observation arg?
        return self.emissions[state][self.observations[time]-1]

    def get_observation_index(self, index):
        """Helper function to do indexing starting 1 instead"""
        return index-1

    def forward(self):
        """ Given an HMM, lambda = (transitions, emissions), and an observation sequence O, 
        determine the likelihood P(O|lambda) 
        observation: array of observations of length T"""

        # i: state_prime
        # j: state
        # t: time
        
        # Initialize alpha = Previous forward path probability. Size # of states by length of observation
        self.alpha = np.zeros((self.state_len, self.observation_len))

        # Initialization: loops through all hidden states
        for state in range(1, self.final_state_index):
            self.alpha[state][self.get_observation_index(1)] = self.transitions[0][state]*self.emission_prob(state, self.get_observation_index(1))

        # Recursion
        for time in range(self.get_observation_index(2), self.observation_len): # Start from the second observation
            for state in range(1, self.final_state_index): # Exclude start and end states here
                for state_prime in range(1, self.final_state_index):
                    self.alpha[state][time] += self.alpha[state_prime][time-1]*self.transitions[state_prime][state]* \
                                    self.emission_prob(state, time) 

        # Termination
        for state in range(1, self.final_state_index):
            self.alpha[self.final_state_index][self.final_observation_index] += self.alpha[state][self.final_observation_index]* \
                                                                    self.transitions[state][self.final_state_index]

        # return self.alpha[self.final_state_index][self.final_observation_index]

    def viterbi(self):
        """ Given an observation sequence O and an HMM, lambda = (transitions, emissions), 
        discover the best hidden state sequence Q """
        path_probability = np.zeros((self.state_len, self.observation_len))

        # Initialize viterbi and backpointer matrix
        viterbi = np.zeros((self.state_len, self.observation_len))
        backpointer = np.zeros((self.state_len,self.observation_len))

        # Initialization
        for state in range(1, final_state_index):
            viterbi[state][1] = self.transitions[0][state]*self.emissions(state,self.get_observation(1))

        # Recursion
        for time in range(self.get_observation_index(2), self.observation_len): # Start from the second observation
            for state in range(1, self.final_state_index): # Exclude start and end states here
                max_viterbi = 0
                max_viterbi_index = (-1, -1)
                for state_prime in range(1, self.final_state_index):
                    current_viterbi = viterbi[state_prime][time-1]*self.transitions[state_prime][state]* \
                                        self.emission_prob(state,self.get_observation_index(time))
                    if (current_viterbi > max_viterbi):
                        max_viterbi = current_viterbi
                        max_viterbi_index = (state_prime, time-1)

                viterbi[state][time] = max_viterbi
                # TODO: Tuples in a numpy array?
                backpointer[state][time] = max_viterbi_index

        # Termination
        max_viterbi = 0
        max_viterbi_index = (-1, -1)
        for state in range(1, self.final_state_index):
            current_viterbi = viterbi[state][self.final_observation_index]* \
                                self.transitions[state][self.final_state_index]
            if (current_viterbi > max_viterbi):
                max_viterbi = current_viterbi
                max_viterbi_index = (state, self.final_observation_index)
        viterbi[self.final_state_index][final_observation_index] = max_viterbi
        backpointer[self.final_state_index][final_observation_index] = max_viterbi_index

        # TODO: Return Backtrace Path

    def backward(self):
        """ Given an HMM, lambda, determine the probability, beta, of seeing the 
            observations from time t+1 to the end, given that we are in state i at time t."""
        
        # Initialize beta
        self.beta = np.zeros((self.state_len, self.observation_len))

        # Initialization: Exclude start and end states
        for state in range(1, self.final_state_index):
            self.beta[state][self.final_observation_index] = self.transitions[state][self.final_state_index]

        # Recursion

        # i: state
        # j: state_prime
        # t: time
        # We want to calculate this numbers back to front
        for time in range(self.final_observation_index-1,self.get_observation_index(1)-1,-1):
            for state in range(1, self.final_state_index): # Exclude start and end states here
                for state_prime in range(1, self.final_state_index):
                    self.beta[state][time] += self.transitions[state][state_prime] *  \
                        self.emission_prob(state_prime,time+1) * \
                        self.beta[state_prime][time+1]
                    # print self.transitions[state][state_prime], "mul", self.emission_prob(state_prime,time+1), "mul", self.beta[state_prime][time+1], "equals", self.beta[state][time]

        # Termination
        for state in range(1, self.final_state_index):
            update = self.transitions[0][state] * self.emission_prob(state,self.get_observation_index(1)) * self.beta[state][self.get_observation_index(1)]
            self.beta[0][self.get_observation_index(1)] += update

        # # TODO: what does the backward algorithm return?
        # return self.beta[0][0]

    # zeta: the probability of being in state i and time t and state j and time t+1
    # for updating self.transitions (A)
    # i: state
    # j: state_prime
    def calc_squiggle(self, state, state_prime, time): 
        # Time should already be adjusted before calling the function
        self.zeta[state][state_prime][time] = (self.alpha[state][time] * self.transitions[state][state_prime] * \
                                                self.emission_prob(state_prime,time+1) * self.beta[state_prime][time+1]) / \
                                                self.alpha[self.final_state_index][self.final_observation_index]

    # gamma: the probability of being in state j at time t
    # for updating self.emissions (B)
    def calc_gamma(self, state, time):
        # Time should already be adjusted before calling the function
        self.gamma[state][time] = self.alpha[state][time] * self.beta[state][time] / self.alpha[self.final_state_index][self.final_observation_index]

    def update_transitions(self, state, state_prime):
        # update transitions (A)
        zeta_sum_num = 0
        zeta_sum_den = 0
        for time in range(self.get_observation_index(1), self.final_observation_index-1):
            zeta_sum_num += self.zeta[state][state_prime][time]
            for state_k in range(1, self.final_state_index):
                zeta_sum_den += self.zeta[state][state_k][time]
        self.transitions[state][state_prime] = zeta_sum_num / zeta_sum_den

    def update_emissions(self, state, v_k):
         # update emissions (B)
        gamma_sum_num = 0
        gamma_sum_den = 0

        # sum over all t for which the observation at time t was v_k
        for time in range(self.get_observation_index(1), self.final_observation_index):
            if self.observations[time] == v_k:
                gamma_sum_num += self.gamma[state][time]
            gamma_sum_den += self.gamma[state][time]

        # We subtract 1 because indexing
        self.emissions[state][v_k-1] = gamma_sum_num / gamma_sum_den

    def baum_welch(self):
        """ Given an observation sequence O and the set of states in the HMM, 
            learn the transitions and emissions of the HMM."""

        # iterate until convergence
        # TODO: Replace this with a while loop
        for i in range(0,1):
            old_A = self.transitions
            old_B = self.emissions
            # expectation step
            self.forward()
            # print "ALPHA"
            # print self.alpha

            self.backward()
            # print "BETA"
            # print self.beta

            for time in range(self.get_observation_index(1), self.observation_len):
                for state in range(1, self.final_state_index):
                    self.calc_gamma(state,time)
            
            for time in range(self.get_observation_index(1), self.final_observation_index):
                for state in range(1, self.final_state_index):
                    self.calc_gamma(state,time)
                    for state_prime in range(1, self.final_state_index):
                        self.calc_squiggle(state,state_prime,time)
            print "GAMMA"
            print self.gamma

            print "ZETA"
            print self.zeta

            # maximization step
            # for state in range(1,self.final_state_index):
            #     for state_prime in range(1,self.final_state_index):
            #         self.update_transitions(state,state_prime)

            # for time in range(self.get_observation_index(1), self.final_observation_index):
            #     for state in range(1, self.final_state_index):
            #         v_k = self.observations[time]
            #         self.update_emissions(state,v_k)
            
            # print "TRANSISTIONS"
            # print self.transitions

            # print "EMISSIONS"
            # print self.emissions
            
            # TODO: how to determine convergence
            # print np.linalg.norm(old_A-self.transitions)
            # print np.linalg.norm(old_B-self.emissions)
                
            # return A, B
            # return self.transitions, self.emissions
            
    def train(self, observations, iterations=10):
        """
        Trains the model and calculates transitions and emissions probabilities
        Input = Array of ice-creams eaten each day
        """
        self.observations = observations
        self.observation_len = len(self.observations)
        self.final_observation_index = self.observation_len-1

        self.zeta = np.zeros((self.state_len, self.state_len, self.observation_len))
        self.gamma = np.zeros((self.state_len, self.observation_len))
        for i in range(0,iterations):
            # TODO: Do the Expectation Maximization step
            self.baum_welch()

if __name__ == '__main__':
    states = [0, 1, 2, 3] # Where 0 = start, 1 = hot, 2 = cold, 3 = final
    # Given ith state, how likely will it transition to jth state
    transitions = np.array([[0.0,0.5,0.5,0.0],[0.0,0.8,0.1,0.1],[0.0,0.1,0.8,0.1],[0.0,0.0,0.0,0.0]])
    # Given ith state, probability of seeing 1, 2, 3 ice-creams.
    emissions = np.array([[0.0,0.0,0.0],[0.1,0.2,0.7],[0.7,0.2,0.1],[0.0,0.0,0.0]])
    observations = [2,3,3,2,3,2,3,2,2,3,1,3,3,1,1,1,2,1,1,1,3,1,2,1,1,1,2,3,3,2,3,2,2] # TODO: Change if necessary
    hmm = HMM(name='Brook', transitions=transitions, emissions=emissions, states=states)

    hmm.train(observations,1)
