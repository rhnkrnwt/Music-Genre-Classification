import math
import time
import numpy as np
from numpy import linalg
import pylab
import random

# Globals
genre = ['pop', 'jazz', 'metal', 'classical']
color = ['-bs', '-rs', '-ms', '-gs']
data = []
train_songs = []
test_songs = []
mfcc = []
genre_delim = 1500
song_n = 15
song_p = 750
genre_pointer = 0
genre_test = 30
plotX = []
plotY = []
# Every song has dimension n x p (song_n x song_p)


class Song:
	def __init__(self, song, genre_code):
		self.data = self.parse(song)
		# self.data = np.asmatrix(self.parse(song))
		# self.data_trans = self.data.T
		self.mean = np.mean(self.data, axis=1)
		self.cov = np.cov(self.data)
		self.cov_inv = linalg.inv(self.cov)
		self.genre = genre_code

	def parse(self, song):
		for i in range(len(song)):
			song[i] = song[i].split(",")
			for j in range(len(song[i])):
				song[i][j] = float(song[i][j])
		return np.array(song)

def KL_divergence(song1, song2):
	# Skipped the log(det(sample.cov_inv)/det(test.sample.cov_inv)) coz core purpose is to calculate distance not 
	dist = -song_n
	dist += np.trace(song2.cov_inv.dot(song1.cov))
	# print(dist)
	temp = (song1.mean - song2.mean)
	dist +=  temp.T.dot(song2.cov_inv).dot(temp)
	return abs(dist/2)

def hellingers_distance(song1, song2):
	diff_mean = np.array((song1.mean - song2.mean))[np.newaxis]
	sum_cov = (song1.cov + song2.cov)/2
	exp_part = (-1/8)*(np.dot(np.dot(diff_mean, linalg.inv(sum_cov)), diff_mean.T))
	exp_part = np.exp(exp_part)
	scale_fac = math.pow((linalg.det(song1.cov) * linalg.det(song2.cov)), 0.25)
	scale_fac /= math.pow(linalg.det(sum_cov), 0.5)
	try:
		return math.sqrt(1 - (scale_fac * exp_part))
	except Exception:
		return 0

def KNN(iterLoop):

	for K in range(1, iterLoop + 1):
		accuracy = [0, 0, 0, 0]
		for test_song in test_songs:
			dist = []
			for train_song in train_songs:
				dist.append((hellingers_distance(test_song, train_song), train_song.genre))
				# dist.append((KL_divergence(test_song, train_song) + KL_divergence(train_song, test_song), train_song.genre))
			dist = sorted(dist)
			count = [0] * (len(genre))
			for j in range(K):
				count[dist[j][1]] += 1
			if count.index(max(count)) == test_song.genre:
				# adding 100 to handle percentage
				accuracy[test_song.genre] += 100
		for genre_pointer in range(len(genre)):
			plotX[genre_pointer].append(K)
			plotY[genre_pointer].append((accuracy[genre_pointer]/genre_test))
			print("Accuracy of", genre[genre_pointer], "is", round(accuracy[genre_pointer]/genre_test) , "%")

if __name__ == '__main__':
	try:
		f = open("../Data/dataset.csv")
		mfcc = f.readlines()
	except FileNotFoundError:
		print("File not found")
	except IOError:
		print("I/O Error")
	finally:
		f.close()
	print("Classifying..")
	for genre_pointer in range(len(genre)):
		plotX.append([])
		plotY.append([])
		genre_total = int(genre_delim/song_n)
		test_indices = random.sample(range(genre_pointer * genre_total, (genre_pointer + 1) * genre_total), genre_test)
		for i in range(genre_delim * genre_pointer, genre_delim * (genre_pointer + 1), song_n):
			if int(i/song_n) in test_indices:
				test_songs.append(Song(mfcc[i : i + song_n], int(i/genre_delim)))
			else:
				train_songs.append(Song(mfcc[i : i + song_n], int(i/genre_delim)))

	start = time.clock()
	KNN(10)
	end = time.clock()
	print("Time taken : ", (end - start))
	print("Plotting..")
	pylab.plot(plotX[0], plotY[0], color[0], label=genre[0])
	pylab.plot(plotX[0], plotY[1], color[1], label=genre[1])
	pylab.plot(plotX[0], plotY[2], color[2], label=genre[2])
	pylab.plot(plotX[0], plotY[3], color[3], label=genre[3])
	
	pylab.xlabel("k")
	pylab.ylabel("Accuracy (%)")
	pylab.legend(loc='best')
	pylab.ylim(-5, 110)

	pylab.show()
	print(plotX, plotY)
