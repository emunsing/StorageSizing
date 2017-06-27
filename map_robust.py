import multiprocessing
import random, sys

# map(njobs, func, iterable, chunksize)


def russianroulette(n):
	if (random.random() < 0.2):
		sys.exit(1)

	return(n**2)


def runFunc(func,data,i,resultQueue):



def robustMap(njobs, func, iterable, chunksize=None)

	if chunksize is not None:
		numChunks = len(data) / chunksize + 1  # Casts to int in division
		chunks = [iterable[i*chunksize:min((i+1)*chunksize,len(iterable))] for i in range(numChunks)]
	else:
		chunks = iterable

	resultQ = multiprocessing.Queue()

	# Create a worker.  As long as the 

	while 
