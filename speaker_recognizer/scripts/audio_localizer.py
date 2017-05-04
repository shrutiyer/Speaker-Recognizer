#!/usr/bin/env python

import rospy

from scipy.io import wavfile
import numpy as np
import math
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
# import matplotlib.pyplot as plt


"""
Predicts the angle from which the audio was received

"""
class Audio_Localizer(object):

	def __init__(self):
		rospy.init_node('localizer')
		rospy.Subscriber('/odom', Odometry, self.process_odom)

		self.sound_speed = 340.29*100 # cm/s
		self.mic_dist = 30 # cm
		self.buffer = 200
		self.angles = []
		
		# angle odometry
		self.angle_curr = 0.0
		self.angle_k = 1
		self.angle_error = None
		self.at_speaker = False
		self.angle_pred = 0.0

		self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

	def calculate_angle_error(self):
		self.angle_error = self.angle_diff(self.angle_pred, self.angle_curr)
		print "angle error: " + str(self.angle_error) + "   angle: " + str(self.angle_curr) + "   goal: " + str(self.angle_pred)

	def angle_diff(self, a, b):
		a = self.angle_normalize(a)
		b = self.angle_normalize(b)
		d1 = a-b
		d2 = 2*math.pi - math.fabs(d1)
		if d1>0:
			d2 *= -1.0
		if (math.fabs(d1)<math.fabs(d2)):
			return d1
		else:
			return d2

	def angle_normalize(self, z):
		return math.atan2(math.sin(z),math.cos(z))

	def process_odom(self, msg):
		orientation = msg.pose.pose.orientation
		orientation_tuple = (orientation.x, orientation.y, orientation.z, orientation.w)
		print orientation_tuple
		angles = euler_from_quaternion(orientation_tuple)
		print angles
		self.angle_curr = angles[2]

	# TODO: callback that listens to audio topic
	def angle_from_audio(self, file_name, chunks):
		[rate, wave] = wavfile.read(file_name)
		raw_0 = wave[:, 0].astype(np.float64)
		raw_1 = wave[:, 1].astype(np.float64)

		for i in range(1, chunks):
			start = i*chunks
			end = (i+1)*chunks

			left = raw_0[start:end]
			right = raw_1[start-self.buffer:end+self.buffer]

			corr_arr = np.correlate(right, left, 'valid')	
			max_index = (len(corr_arr)/2)-np.argmax(corr_arr) 
			time_d = max_index/float(rate)
			signal_dist = time_d*self.sound_speed

			if (signal_dist != 0 and abs(signal_dist)<=self.mic_dist):
				angle = math.degrees(math.asin( signal_dist / self.mic_dist))
				self.angles.append(angle)
		
		a = np.array(self.angles)
		hist, bins = np.histogram(a, bins=10)

		# width = 0.7 * (bins[1] - bins[0])
		# center = (bins[:-1] + bins[1:]) / 2
		# plt.bar(center, hist, align='center', width=width)
		# plt.xlabel('Angle (degrees)', fontsize=16)
		# plt.show()

		index = np.argmax(hist)
		self.angle_pred = bins[index]

		print self.angle_pred

	def localize(self):
		self.at_speaker = self.calculate_angle_error()
		if self.at_speaker:
			self.stop()
		else:
			twist = self.calculate_twist()
			self.pub.publish(twist)

	def calculate_twist(self):
		twist = Twist()
		twist.angular.z = self.angle_error * self.angle_k
		return twist

	def stop(self):
		twist = Twist()
		self.pub.publish(twist)

	def run(self):
		r = rospy.Rate(10)

		while not rospy.is_shutdown():
			try:
				self.localize()
				r.sleep()
			except rospy.exceptions.ROSTimeMovedBackwardsException:
				print "Time went backwards. Carry on."

if __name__ == '__main__':
	al = Audio_Localizer()
	al.run()