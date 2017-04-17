#!/usr/bin/env python

import rospy

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from scipy.cluster.vq import kmeans2
import numpy as np


class HMM(object):

    def __init__(self, name=None, transitions=None, emissions=None):

        self.name = None
        self.transitions = transitions # Matrix of transision probabilities from one state to another. Size # of states by # of states
        self.emissions = emissions # Matrix of emission probabilities. Size # of states by # of observation symbol
        self.states = None
        self.observations = None

    def transition_prob(self, i_state, j_state):
        # TODO: Include start probabilities in here
        if not(i_state in self.states and j_state in self.states):
            return None
        return self.transitions[i_state][j_state]

    def emission_prob(self, state, observation):
        if not(state in self.states and observation in self.observations):
            return None
        return self.emissions[state][observation]

    def forward(self, observation):
        """ Given an HMM, lambda = (transitions, emissions), and an observation sequence O, 
        determine the likelihood P(O|lambda) 
        observation: array of observations of length T"""
        
        observation_len = len(observation)
        state_len = len(self.states)

        # Initialize alpha = Previous forward path probability. Size # of states by length of observation
        alpha = [[x for x in range(observation_len)] for y in range(state_len)]

        # Initialization
        for state in self.states:
            alpha[0][state] = self.transition_prob(0,state)*self.emissions(state,observation[0])

        # Recursion
        for time in range(1, observation_len):

        # Termination

    def viterbi(self, observation):
        """ Given an observation sequence O and an HMM, lambda = (transitions, emissions), 
        discover the best hidden state sequence Q """
        pass

    def forward_backward(self):
        """ Given an observation sequence O and the set of states in the HMM, 
            learn the transitions and emissions of the HMM."""
        pass


class Recognizer(object):

    def __init__(self, audio_topic):
        rospy.init_node('recognizer')

        # TODO: AudioData msg?
        # self.sub = rospy.Subscriber(audio_topic, AudioData, self.process_audio)

    def process_audio(self):
        """ Takes in a wav file and outputs a 13 dimension MFCC vector
        """
        # (rate, sig) = msg
        # TODO: how to translate microphone audio to correct format?

        (rate, sig) = wav.read("english.wav")
        mfcc_feat = mfcc(sig, rate)

        # TODO: Figure out if we are doing clustering correctly
        mfcc_centroids = kmeans2(mfcc_feat[:,:], 1)[0]


    def run(self):
        """ Main run function """
        
        r = rospy.Rate(50) # 20 ms samples
        
        while not rospy.is_shutdown():  
            try:
                self.process_audio()
                r.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                print "Time went backwards. Carry on."


if __name__ == '__main__':
    recognizer = Recognizer("/audio/audio")
    recognizer.run()