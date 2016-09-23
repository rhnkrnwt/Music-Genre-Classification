import numpy as np
from numpy import linalg

# Globals
genre = ['pop', 'jazz', 'metal', 'classic']
data = []
songs = []
genre_delim = 1500
song_n = 15
song_p = 750
# Every song has dimension n x p  

class Song:
	def __init__(self, song, genre_code):
		self.data = self.parse(song)
		# self.data = np.asmatrix(self.parse(song))
		# self.data_trans = self.data.T
		self.mean = np.mean(self.data, axis=0)
		self.cov = np.cov(self.data.T)
		# self.inv = linalg.inv(self.cov)
		self.genre = genre[genre_code]

	def parse(self, song):
		for i in range(len(song)):
			song[i] = song[i].split(",")
			for j in range(len(song[i])):
				song[i][j] = float(song[i][j])
		song = np.array(song)
		return song


if __name__ == '__main__':
	try:
		f = open("../Data/dataset.csv")
		songs = f.readlines()
	except FileNotFoundError:
		print("File not found")
	except IOError:
		print("I/O Error")
	finally:
		f.close()
	for i in range(0, len(songs), song_n):
		songs[i] = Song(songs[i : i + song_n], int(i/genre_delim))
	print(len(songs[0].cov[0]))
