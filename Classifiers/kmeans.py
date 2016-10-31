import math
import time
import numpy as np
from numpy import linalg
import pylab
from random import randint

# Globals
genre = ['pop', 'jazz', 'metal', 'classical']
data = []
songs = []
mfcc = []
genre_delim = 1500
song_n = 15
song_p = 750
# Every song has dimension n x p (song_n x song_p)


class Song:
	def __init__(self, song_data, genre_code, dataParsed):
		if not dataParsed:
			self.data = self.parse(song_data)
		else:
			self.data = song_data
		self.genre = genre_code
		self.cluster = None
		self.mean = np.mean(self.data, axis=1)
		self.cov = np.cov(self.data)
		self.calculate()
		# cluster can be zero, one, two or three

	def calculate(self):
		# self.data = np.asmatrix(self.parse(song))
		# self.data_trans = self.data.T
		self.cov_inv = linalg.inv(self.cov)

	def reinit(self):
		self.data = self.data * 0
		self.mean = self.mean * 0
		self.cov = self.cov * 0
	
	def parse(self, song_data):
		for i in range(len(song_data)):
			song_data[i] = song_data[i].split(",")
			for j in range(len(song_data[i])):
				song_data[i][j] = float(song_data[i][j])
		return np.array(song_data)

def KL_divergence(song1, song2):
	# Skipped the log(det(sample.cov_inv)/det(test.sample.cov_inv)) coz core purpose is to calculate distance not 
	dist = -song_n
	dist += np.trace(song2.cov_inv.dot(song1.cov))
	# print(dist)
	temp = (song1.mean - song2.mean)
	dist +=  temp.T.dot(song2.cov_inv).dot(temp)
	return abs(dist/2)

def checkThresh(deltaMean):

	meanThresh = 0.95 * np.ones(song_n)
	for i in range(len(genre)): 
		for j in range(song_n):
			if meanThresh[j] > abs(deltaMean[i][j]):
				return True
	return False


def Kmeans(testLoopSize):
	deltaMean = [None] * len(genre)
	# deltaCov = [None] * len(genre)
	total = np.zeros((len(genre), len(genre)))
	song_centroid = []
	for loopCounter in range(testLoopSize):
		for i in range(len(genre)):
			temp = int(randint(i * len(songs)/len(genre), (i + 1) * len(songs)/len(genre) - 1))
			song_centroid.append(Song(songs[temp].data, i, True))
			song_centroid[i].cluster = i
		distance = [0]*len(genre)

		loop = True
		while loop:
			for song in songs:
				for i in range(len(genre)):
					distance[i] = KL_divergence(song_centroid[i], song) + KL_divergence(song, song_centroid[i])
				song.cluster = distance.index(min(distance))

			for i in range(len(genre)):
				deltaMean[i] = song_centroid[i].mean
				song_centroid[i].reinit()
			
			counter = [0]*len(genre)
			
			for song in songs:
				song_centroid[song.cluster].mean += song.mean
				song_centroid[song.cluster].cov += song.cov
				counter[song.cluster] += 1

			for i in range(len(genre)):
				song_centroid[i].mean /= counter[i]
				song_centroid[i].cov /= counter[i]
				for j in range(len(song_centroid[i].mean)):
					deltaMean[i][j] = deltaMean[i][j]/song_centroid[i].mean[j]
				try:
					song_centroid[i].calculate()
				except:
					print('Error in Classification')
			loop = checkThresh(deltaMean)
		for song in songs:
			total[song.genre][song.cluster] += 1
	print(total/testLoopSize)
	

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
	for i in range(0, len(mfcc), song_n):
		songs.append(Song(mfcc[i : i + song_n], int(i/genre_delim), False))
	print("Starting Clustering...")
	
	
	#Kmeans function takes in one parameter and that is number of instances over which observations should be averaged upon.
	Kmeans(10)
