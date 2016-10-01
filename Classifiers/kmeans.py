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
	def __init__(self, song, genre_code):
		self.data = self.parse(song)
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

	def zero_out(self):
		self.data = self.data * 0
		self.mean = self.mean * 0
		self.cov = self.cov * 0
	
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

def Kmeans(epochs):
	#1
	song_centroid = [None]*len(genre)
	for i in range(len(genre)):
		temp = int(randint(i * genre_delim, (i + 1) * genre_delim - 1)/song_n)
		# print(temp)
		song_centroid[i] = songs[temp]
		song_centroid[i].cluster = i
	distance = [0]*len(genre)
	#2
	while epochs:
		for song in songs:
			for i in range(len(genre)):
				distance[i] = KL_divergence(song_centroid[i], song) + KL_divergence(song, song_centroid[i])
			song.cluster = distance.index(min(distance))

		counter = [0]*len(genre)
		for song in songs:
			counter[song.cluster] += 1

		for i in range(len(genre)):
			print(counter[i], end=' ')
		
		for i in range(len(genre)):
			# print(song_centroid[i].mean, end=' ')
			song_centroid[i].zero_out()
		
		print()
		# print("Updating centroid : ")
		
		for song in songs:
			song_centroid[song.cluster].mean += song.mean
			song_centroid[song.cluster].cov += song.cov

		for i in range(len(genre)):
			song_centroid[i].mean *= song_n / genre_delim
			song_centroid[i].cov *= song_n / genre_delim
			# print(song_centroid[i].cov)
			song_centroid[i].calculate()
		epochs-=1
	

	

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
		songs.append(Song(mfcc[i : i + song_n], int(i/genre_delim)))
	print("Starting Clustering...")
	Kmeans(2)