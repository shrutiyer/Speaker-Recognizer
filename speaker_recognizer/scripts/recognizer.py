#!/usr/bin/env python

import rospy

from python_speech_features import mfcc
import scipy.io.wavfile as wav


class HMM(object):

    def __init__(self, name=None, transitions=None, emissions=None):

        self.name = None
        self.transitions = None
        self.emissions = None


    def forward(self):
        """ Given an HMM, lambda = (transitions, emissions), and an observation sequence O, 
        determine the likelihood P(O|lambda) """
        pass

    def viterbi(self):
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
        self.sub = rospy.Subscriber(audio_topic, AudioData, self.process_audio)

    def process_audio(self, msg):
        print "Processing audio."

        # (rate, sig) = msg
        # TODO: how to translate microphone audio to correct format?

        (rate, sig) = wav.read("hello.wav")
        mfcc_feat = mfcc(sig, rate)
        d_mfcc_feat = delta(mfcc_feat, 2)
        fbank_feat = logfbank(sig, rate)

        print (fbank_feat[1:3, :]) # rows (frames), columns (features)

        # TODO: kmeans clustering on features to initialize hmm parameters


    def run(self):
        """ Main run function """
        
        r = rospy.Rate(50) # 20 ms samples
        
        while not rospy.is_shutdown():
            
            try:
                r.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                print "Time went backwards. Carry on."


if __name__ == '__main__':
    recognizer = Recognizer("/audio/audio")
    recognizer.run()