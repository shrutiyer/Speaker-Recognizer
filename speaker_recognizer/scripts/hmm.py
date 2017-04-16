#!/usr/bin/env python

import rospy

class HMM(object):

    def __init__(self, transitions=None, emissions=None):

        # Initialize ROS node
        rospy.init_node('hmm')

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


    def run(self):
        """ Main run function """
        
        r = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            
            try:
                r.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                print "Time went backwards. Carry on."


if __name__ == '__main__':
    hmm = HMM()
    hmm.run()