import multiprocessing
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

import pandas as pd
import numpy as np
import scipy.sparse as sps1
from copy import deepcopy
from datetime import *
import dateutil
from dateutil import parser, relativedelta
import pytz
import sys, os, time
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)


from simulationFunctions import *


print("Current environment directory:" + sys.prefix)
print("System version: "+sys.version)


def computePriceSweep(thisSlice,chunkNum, qpid, qresult):
	myLength = thisSlice.shape[1]

	myPID = multiprocessing.current_process().pid

	# Add our process id to the list of process ids
	while qpid.empty():
		time.sleep(0.01)
	pidList = qpid.get()
	pidList.append(myPID)
	qpid.put(pidList)

	storagePriceSet = np.arange(1.0,20.1,0.25)
	storagePriceSet = np.arange(1.0,3.1,1)
	eff_round = 0.9  # Round-trip efficiency
	E_min = 0
	E_max = 1

	# Endogenous parameters; calculated automatically
	(eff_in, eff_out) = [np.sqrt(eff_round)] *2  # Properly account for the round trip efficiency of storage
	P_max = E_max/eff_in # Max discharge power at grid intertie, e.g max limit for C_i
	P_min = -1*E_max/eff_out # Max charge power at grid intertie, e.g. min D_i

	# Create a set of zero-filled dataframes for storing results
	resultIndex = pd.MultiIndex.from_product([thisSlice.index,['size','kwhPassed','profit']])
	results = pd.DataFrame(index = resultIndex, columns=storagePriceSet)

	# Create clean problem matrices - needs efficiency and length!
	(A, b, A_eq, b_eq) = createABMatrices(myLength, delta_T, eff_in, eff_out, P_min, P_max, E_min, E_max)
	# Define model:
	coolStart = CyClpSimplex()
	coolStart.logLevel = 0
	x_var = coolStart.addVariable('x',myLength*3+2)
	# Add constraints to model:
	coolStart += A * x_var <= b.toarray()
	coolStart += A_eq * x_var == b_eq.toarray()


	newIDs = [] # this holds the names of nodes which we've recently processed

	def dumpResults():
		results.dropna(axis=0,how='all').to_csv('Data/priceSweepResults_pid'+str(myPID)+'temp.csv' )

		# Make sure that the results queue hasn't been checked out by somebody else
		while qresult.empty():
			time.sleep(0.01)
		resultList = qresult.get()

		resultList[chunkNum] = resultList[chunkNum] + newIDs
		qresult.put(resultList)

	for i in range(thisSlice.shape[0]):
		## Set up prices
		myNodeName = thisSlice.index[i]
		energyPrice = thisSlice.loc[myNodeName,:] / 1000.0 # Price $/kWh as array

		# if (np.random.random(1)[0] < 0.05):
		# 	sys.exit(1) # Randomly kill the process

		c = np.concatenate([[0.0],[0.0]*(myLength+1),energyPrice,energyPrice],axis=0)  #placeholder; No cost for storage state; charged for what we consume, get paid for what we discharge
		#[[storagePricePlaceholder],[0]*(myLength+1),myPrice,myPrice],axis=1)
		c_clp = CyLPArray(c)

		sweepStartTime = time.time()

		for myStoragePrice in storagePriceSet:
			c_clp[0] = myStoragePrice * simulationYears
			coolStart.objective = c_clp * x_var

			# Run the model
			coolStart.primal()

			# Results- Rows are Nodes, indexed by name. Columns are Storage price, indexed by price
			x_out = coolStart.primalVariableSolution['x']

			results.loc[(myNodeName,'size'),     myStoragePrice] = x_out[0]
			results.loc[(myNodeName,'profit'),   myStoragePrice] = np.dot(-c, x_out)
			results.loc[(myNodeName,'kwhPassed'),myStoragePrice] = sum(x_out[2+myLength : 2+myLength*2]) * eff_in # Note: this is the net power pulled from the grid, not the number of cycles when the system is unconstrained

		storagePriceSet = storagePriceSet[::-1] # Reverse the price set so that we start from the same price for the next node to make warm-start more effective

		newIDs.append(myNodeName)

		if (i+1)%5==0:
			dumpResults()  # Write the results to file, and then also update the process Queue with the 
			newIDs = []

	# The results should now be all done!
	dumpResults()
	# results.to_csv('Data/efficiencyResults_pid'+str(myPID)+'temp.csv' )


if __name__ == '__main__':


	try:
		fname = os.environ['INPUTFILE']
	except KeyError:
		fname = "inputData/pricedata_LMP_100.csv" # Only 100 nodes    

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

	## CUSTOM START/END DATE
	#startDate = parser.parse('01/01/12 00:00')  # year starts at 2013-01-01 00:00:00
	#endDate = parser.parse('12/31/12 23:00')  # year ends at 2013-12-31 23:00:00
	#startDate = pytz.timezone('America/Los_Angeles').localize(startDate).astimezone(pytz.utc)
	#endDate = pytz.timezone('America/Los_Angeles').localize(endDate).astimezone(pytz.utc)

	# ## FULL DATASET
	startDate = APNode_Prices.columns.values[ 0].astype('M8[m]').astype('O') # Convert to datetime, not timestamp
	endDate   = APNode_Prices.columns.values[-1].astype('M8[m]').astype('O')
	startDate = pytz.utc.localize(startDate)
	endDate   = pytz.utc.localize(endDate)

	timespan = relativedelta.relativedelta(endDate +timestep, startDate)
	simulationYears = timespan.years + timespan.months/12. + timespan.days/365. + timespan.hours/8760.  # Leap years will be slightly more than a year, and that's ok.

	startNode = 0
	try:
		stopNode = int(os.environ['STOPNODE'])
	except KeyError:
		stopNode  = 15 # if set to zero, then will loop through all nodes
	if ((stopNode == 0)|(stopNode > APNode_Prices.shape[0])): stopNode = APNode_Prices.shape[0]
		
	APNode_Prices = APNode_Prices.ix[startNode:stopNode,startDate:endDate]
	# someResults = computePriceSweep(thisSlice)

	# np.random.seed([1])

	j = min(multiprocessing.cpu_count(),10)
	chunksize = (APNode_Prices.shape[0]/j)+1  # Utilize casting as an int
	slices = [df for g,df in APNode_Prices.groupby(np.arange(APNode_Prices.shape[0])//chunksize)]

	# Create empty Queues
	pidq = multiprocessing.Queue() # this will store a list of all the process ids which have been spawned, in order to collect the results
	pidq.put([])

	processedq = multiprocessing.Queue() # This will store a dictionary mapping the chunks to the keys which have been mapped already 
	processedq.put({k: [] for k in range(j)})

	processDict = dict()    # This maps chunk numbers to the current handler process
	sliceLength = [df.shape[0] for df in slices]

	print("Beginning to spawn processes...")

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
		time.sleep(1)
	pidList = pidq.get()
	print(pidList)

	# Cleanup: Take all of the temporary files and merge them into a single Dataframe, then save that.
	#   Once we're done, remove the files.

	resultDfList = [pd.read_csv('Data/priceSweepResults_pid'+str(myPID)+'temp.csv', header=0,index_col=0) for myPID in pidList]

	joinedResults = pd.concat(resultDfList).sort_index()
	joinedResults.to_csv('Data/priceSweepResults.csv')

	if type(joinedResults.index)==pd.indexes.base.Index:
		joinedResults = joinedResults.set_index('Unnamed: 1',append=True).sort_index()

	[os.remove('Data/priceSweepResults_pid'+str(myPID)+'temp.csv') for myPID in pidList]

	sizeDf   = joinedResults.loc[(slice(None),'size'),:].reset_index(level=1,drop=True)
	profitDf = joinedResults.loc[(slice(None),'profit'),:].reset_index(level=1,drop=True)
	cycleDf  = joinedResults.loc[(slice(None),'kwhPassed'),:].reset_index(level=1,drop=True)

	sizeDf.to_csv('Data/VaryingPrices_StorageSizing_v2.csv')
	profitDf.to_csv('Data/VaryingPrices_StorageProfits_v2.csv')
	cycleDf.to_csv('Data/VaryingPrices_StorageCycles_v2.csv')
	print("All done!")

