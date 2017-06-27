import multiprocessing
import sys, os
import time
import pandas as pd
import numpy as np
import dateutil
from dateutil import parser, relativedelta

print("Current environment directory:" + sys.prefix)
print("System version: "+sys.version)


## Testing Multiprocessing

# Want one process that works well, and another that fails

def deadProcess():
	print("Bad process at %s" % multiprocessing.current_process().pid)
	sys.exit(1)

def liveProcess():
	print("Good process at %s" % multiprocessing.current_process().pid)
	time.sleep(2)


# Chunk the dataset into blocks

# Take the chunks, and for each chunk spawn a worker

# Basically want to monitor each chunk, and make sure that it gets handled
# Create a queue mapping chunks to process ids
# Create a queue mapping process ids to current results

# Loop through the chunks.
#  If the worker is dead:
#     check whether the chunk is also done based on comparison of the result index and the assignment index
#     If the worker ended prematurely,
#        Start a new process which is assigned the remainder of the chunk 


# Process is spawned:

# Write process id to the Queue mapping chunks to pids

def computePriceSweep(thisSlice,chunkNum, qpid, qresult):
	myPID = multiprocessing.current_process().pid

	# Add our process id to the list of process ids
	while qpid.empty():
		time.sleep(0.01)
	pidList = qpid.get()
	pidList.append(myPID)
	qpid.put(pidList)

	storagePriceSet = np.arange(1.0,20.1,0.25)
	results = pd.DataFrame(index = thisSlice.index, columns=storagePriceSet)

	newIDs = []

	def dumpResults():
		results.dropna(axis=0,how='all').to_csv('Data/efficiencyResults_pid'+str(myPID)+'temp.csv' )

		# Make sure that the results queue hasn't been checked out by somebody else
		while qresult.empty():
			time.sleep(0.01)
		resultList = qresult.get()

		resultList[chunkNum] = resultList[chunkNum] + newIDs
		qresult.put(resultList)


	for i in range(thisSlice.shape[0]):		# Pretend we're doing a ton of processing
		myNodeName = thisSlice.index[i]
		# Hella complicated calculations ... or at least a simulation of such
		if (np.random.random(1)[0] < 0.05):
			sys.exit(1) # Randomly kill the process

		nodeResults = storagePriceSet + i
		results.loc[myNodeName,:] = nodeResults
		time.sleep(0.1)

		newIDs.append(myNodeName)

		if (i+1)%3==0:
			dumpResults()  # Write the results to file, and then also update the process Queue with the 
			newIDs = []

	# The results should now be all done!
	dumpResults()
	# results.to_csv('Data/efficiencyResults_pid'+str(myPID)+'temp.csv' )


if __name__ == '__main__':

	np.random.seed([1])

	fname = 'inputData/priceData_LMP_100_short.csv'
	APNode_Prices = pd.read_csv( fname, header=0,index_col=0)#,nrows=10)
	APNode_Prices.columns = pd.DatetimeIndex(APNode_Prices.columns,tz=dateutil.tz.tzutc())  # Note: This will be in UTC time. Use .tz_localize(pytz.timezone('America/Los_Angeles')) if a local time zone is desired- but note that this will 
	timestep = relativedelta.relativedelta(APNode_Prices.columns[2],APNode_Prices.columns[1])
	delta_T = timestep.hours  # Time-step in hours

	## Deal with NaN prices
	# Drop nodes which are above a cutoff
	goodNodes = (APNode_Prices.isnull().sum(axis=1) < (0.02 * APNode_Prices.shape[1])) # True if node is less than x% NaN values
	APNode_Prices = APNode_Prices[goodNodes]
	# Interpolate remaining NaNs
	APNode_Prices.interpolate(method='linear',axis=1)
	print("Finished Loading Data")
	sys.stdout.flush()

	j = min(multiprocessing.cpu_count(),10)
	j = 5
	chunksize = (APNode_Prices.shape[0]/j)+1  # Utilize casting as an int
	slices = [df for g,df in APNode_Prices.groupby(np.arange(APNode_Prices.shape[0])//chunksize)]


	# data = range(13)
	# chunkSize = 7
	# numChunks = len(data) / chunkSize + 1  # Casts to int in division
	# slices = [data[i*chunkSize:min((i+1)*chunkSize,len(data))] for i in range(numChunks)]

	# Create empty Queues
	pidq = multiprocessing.Queue() # this will store a list of all the process ids which have been spawned, in order to collect the results
	pidq.put([])

	processedq = multiprocessing.Queue() # This will store a dictionary mapping the chunks to the keys which have been mapped already 
	processedq.put({k: [] for k in range(j)})

	processDict = dict()    # This maps chunk numbers to the current handler process
	sliceLength = [df.shape[0] for df in slices]

	for i in range(j):
		myProcess = multiprocessing.Process(target=computePriceSweep,args=(slices[i],i,pidq,processedq))
		myProcess.start()
		processDict[i] = myProcess

	stillRunningFlag = True


	while stillRunningFlag:
		stillRunningFlag = False
		resultList = processedq.get()
		processedq.put(resultList)

		print("Checking Loop")
		for i in range(j):			
			alive = processDict[i].is_alive()
			finishedChunk = (len(resultList[i]) >= sliceLength[i])

			print("Chunk %s:\t Computed %s of %s. Finished? %s \t Still alive? %s"%
				  (i, len(resultList[i]),sliceLength[i],finishedChunk, alive) )

			if not(alive) and not(finishedChunk):
				# Define a slice with the remaining 
				incomplete = set(slices[i].index.values) - set(resultList[i])

				thisSlice = slices[i].loc[incomplete,:]
				myProcess = multiprocessing.Process(target=computePriceSweep,args=(thisSlice,i,pidq,processedq))
				print("Process %s was dead; spawning new process to finish %s entries"%(i,len(incomplete)))

				myProcess.start()
				processDict[i] = myProcess
				alive = True


			stillRunningFlag += alive
		time.sleep(0.25)
	pidList = pidq.get()
	print(pidList)

	# Cleanup: Take all of the temporary files and merge them into a single Dataframe, then save that.
	#   Once we're done, remove the files.

	resultDfList = [pd.read_csv('Data/efficiencyResults_pid'+str(myPID)+'temp.csv', header=0,index_col=0) for myPID in pidList]

	resultDf = pd.concat(resultDfList).sort()
	resultDf.to_csv('Data/efficiencyResults.csv')

	[os.remove('Data/efficiencyResults_pid'+str(myPID)+'temp.csv') for myPID in pidList]

