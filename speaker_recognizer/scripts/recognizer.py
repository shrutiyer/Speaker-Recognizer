#!/usr/bin/env python

import rospy

from python_speech_features import mfcc, delta, logfbank
import scipy.io.wavfile as wav
from scipy.cluster.vq import kmeans2, whiten, vq
import numpy as np


class HMM(object):

    def __init__(self, name=None, transitions=None, emissions=None, observations=None):

        self.name = None
        self.transitions = transitions # Matrix of transision probabilities from one state to another. Size # of states by # of states
        self.emissions = emissions # Matrix of emission probabilities. Size # of states by # of observation symbol
        self.states = None # Includes start (q0) and end (qF) states. Along with N hidden states
        self.observations = observations

        self.observation_len = len(self.observations)
        self.state_len = len(self.states)

        self.final_state_index = state_len-1
        self.final_observation_index = observation_len-1

        self.alpha = None
        self.beta = None
        self.zeta = None
        self.gamma = None

    def transition_prob(self, i_state, j_state):
        # TODO: Include start probabilities in here
        if not(i_state in self.states and j_state in self.states):
            return None
        return self.transitions[i_state][j_state]

    def emission_prob(self, state, observation): # TODO: observation arg?
        if not(state in self.states and observation in self.observations):
            return None
        return self.emissions[state][observation]

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

        # Initialization
        for state in range(1, self.final_state_index):
            self.alpha[state][1] = self.transition_prob(0,state)*self.emissions[state][self.get_observation_index(1)]

        # Recursion
        for time in range(1, self.observation_len): # Start from the second observation
            for state in range(1, self.final_state_index): # Exclude start and end states here
                for state_prime in range(1, self.final_state_index):
                    self.alpha[state][time] += self.alpha[state_prime][time-1]*self.transitions[state_prime][state]* \
                                    self.emissions[state][self.get_observation_index(time)] 

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
            viterbi[state][1] = self.transition_prob(0,state)*self.emissions(state,self.get_observation(1))

        # Recursion
        for time in range(1, self.observation_len): # Start from the second observation
            for state in range(1, self.final_state_index): # Exclude start and end states here
                max_viterbi = 0
                max_viterbi_index = (-1, -1)
                for state_prime in range(1, self.final_state_index):
                    current_viterbi = viterbi[state_prime][time-1]*self.transitions[state_prime][state]* \
                                        self.emissions[state][self.get_observation_index(time)]
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

        # Initialization
        for state in range(1, self.final_state_index):
            self.beta[state][self.final_observation_index] = self.transition_prob(state, self.final_state_index)

        # Recursion

        # i: state
        # j: state_prime
        # t: time
        for time in range(1, self.observation_len): # Start from the second observation
            for state in range(1, self.final_state_index): # Exclude start and end states here
                for state_prime in range(1, self.final_state_index):
                    self.beta[state][time] += self.transitions[state][state_prime] *  \
                        self.emissions[state_prime][self.get_observation_index(time+1)] * \
                        self.beta[state_prime][time+1]

        # Termination
        for state in range(1, self.final_state_index):
            update = self.transitions[0][state] * self.emissions[state][self.get_observation_index(1)] * self.beta[state][1]
            self.alpha[self.final_state_index][self.final_observation_index] += update
            self.beta[0][self.get_observation_index(1)] += update

        # # TODO: what does the backward algorithm return?
        # return self.beta[0][0]

    # zeta: the probability of being in state i and time t and state j and time t+1
    # for updating self.transitions (A)
    # i: state
    # j: state_prime
    def squiggle(self, state, state_prime, time): 
        self.zeta = (self.alpha[state][time] * self.transitions[state][state_prime] * \
            self.emissions[state_prime][self.get_observation_index(time+1)] * \
            self.beta[state_prime][time+1]) / self.alpha[self.final_state_index][self.final_observation_index]

    # gamma: the probability of being in state j at time t
    # for updating self.emissions (B)
    def gamma(self, state, time):
        self.gamma = self.alpha[state][time] * self.beta[state][time] / self.alpha[self.final_state_index][self.final_observation_index]

    def update_transitions(self, state, state_prime):
        # update transitions (A)
        zeta_sum_num = 0
        zeta_sum_den = 0
        for time in range(1, self.final_observation_index-1):
            zeta_sum_num += self.zeta[state][state_prime]
            for state_k in range(1, self.final_state_index):
                zeta_sum_den += self.zeta[state][state_k]

        self.transitions[state][state_prime] = zeta_sum_num / zeta_sum_den

    def update_emissions(self, state, v_k):
         # update emissions (B)
        gamma_sum_num = 0
        gamma_sum_den = 0

        # sum over all t for which the observation at time t was v_k
        for time in range(1, self.final_observation_index):
        # TODO: how to initialize self.observations?
        if self.observations[time] == v_k:
            gamma_sum_num += self.gamma[state, time]
        gamma_sum_den += self.gamma[state, time]

        self.emissions[v_k][state] = gamma_sum_num / gamma_sum_den

    def baum_welch(self):
        """ Given an observation sequence O and the set of states in the HMM, 
            learn the transitions and emissions of the HMM."""
        
        # initialize A and B - TODO: how?

        # iterate until convergence - TODO: how to determine convergence?
            # expectation step
            # maximization step

        # return A, B

class Recognizer(object):

    def __init__(self, audio_topic):
        rospy.init_node('recognizer')
        self.codebook = None

        # TODO: AudioData msg?
        # self.sub = rospy.Subscriber(audio_topic, AudioData, self.process_audio)


    def process_audio(self, isTraining):
        """ Takes in a wav file and outputs labeled observations of the audio
            isTraining: bool that is true if the model is being trained
        """
        # (rate, sig) = msg
        # TODO: how to translate microphone audio to correct format?

        (rate, sig) = wav.read("english.wav")
        # MFCC Features. Each row corresponds to MFCC for a frame
        mfcc_feat = mfcc(sig, rate)

        # Normalize the features
        whitened = whiten(mfcc_feat)

        if (isTraining):
            # Create a codebook and labeled observations
            self.codebook, labeled_obs = kmeans2(data=whitened, k=3)
        else:
            labeled_obs = vq(mfcc_feat, self.codebook)

        return labeled_obs

    def run(self):
        """ Main run function """
        
        r = rospy.Rate(50) # 20 ms samples
        
        while not rospy.is_shutdown():  
            try:
                # self.process_audio(isTraining=True)
                print "Running"
                r.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                print "Time went backwards. Carry on."


if __name__ == '__main__':
    recognizer = Recognizer("/audio/audio")
    recognizer.run()
    hmm = HMM()
    hmm.forward()
