import math
import time
import numpy as np
from numpy import linalg
import pylab

# Globals
genre = ['pop', 'jazz', 'metal', 'classical']
color = ['-bs', '-rs', '-ms', '-gs']
data = []
songs = []
mfcc = []
genre_delim = 1500
song_n = 15
song_p = 750
genre_pointer = 0
genre_train = 30
tester = [genre_pointer * genre_delim, genre_pointer * genre_delim + genre_train * song_n] 
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

def KNN():
	plotX.append([])
	plotY.append([])
	for k in range():
		plotX[genre_pointer].append(k)
		accuracy = 0
		for i in range(genre_train):
			temp = genre_pointer * genre_delim + i * song_n
			s = Song(mfcc[temp : temp + song_n], genre_pointer)
			dist = [float("inf")]
			out = []
			for song in songs:
				temp = KL_divergence(s, song) + KL_divergence(song, s)
				for j in range(k):
					if temp < dist[j]:
						dist.insert(j, temp)
						out.insert(j, song.genre)
						break
			count = [0] * (len(genre))
			for j in range(k):
				count[out[j]] += 1
			if count.index(max(count)) == s.genre:
				accuracy += 1
		# print("Accuracy in classifying", genre[s.genre], ":", round(100*accuracy/genre_train), "%")
		plotY[genre_pointer].append(round(100*accuracy/genre_train))

	

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
	for genre_pointer in range(len(genre)):
		tester = [genre_pointer * genre_delim, genre_pointer * genre_delim + genre_train * song_n] 
		for i in range(0, len(mfcc), song_n):
			if i < tester[0] or i >= tester[1]:
				songs.append(Song(mfcc[i : i + song_n], int(i/genre_delim)))
		# start = time.clock()
		KNN()
	
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
		# end = time.clock()
	# print("Time taken : ", end - start)
