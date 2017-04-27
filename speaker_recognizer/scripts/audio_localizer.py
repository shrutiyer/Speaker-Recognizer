from scipy.io import wavfile
import numpy as np
import math
# import matplotlib.pyplot as plt

sound_speed = 340.29*100 # cm/s
mic_dist = 30 # cm
buf = 200

def audio_chunks(file_name, chunks):
	[rate, wave] = wavfile.read(file_name)
	raw_0 = wave[:, 0].astype(np.float64)
	raw_1 = wave[:, 1].astype(np.float64)

	angles = []

	for i in range(1, chunks):
		start = i*chunks
		end = (i+1)*chunks

		left = raw_0[start:end]
		right = raw_1[start-buf:end+buf]

		corr_arr = np.correlate(right, left, 'valid')	
		max_index = (len(corr_arr)/2)-np.argmax(corr_arr) 
		time_d = max_index/float(rate)
		signal_dist = time_d*sound_speed

		if (signal_dist != 0 and abs(signal_dist)<=mic_dist):
			angle = math.degrees(math.asin( signal_dist / mic_dist))
			angles.append(angle)
	
	a = np.array(angles)
	hist, bins = np.histogram(a, bins=10)

	# width = 0.7 * (bins[1] - bins[0])
	# center = (bins[:-1] + bins[1:]) / 2
	# plt.bar(center, hist, align='center', width=width)
	# plt.xlabel('Angle (degrees)', fontsize=16)
	# plt.show()

	index = np.argmax(hist)
	print "Angle", bins[index]

audio_chunks('../data/test/test1.wav', 5000)