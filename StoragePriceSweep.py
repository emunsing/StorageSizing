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
import sys


from simulationFunctions import *

import sys  # This is necessary for printing updates within a code block, via sys.stdout.flush()
import time # Use time.sleep(secs) to sleep a process if needed

print("Current environment directory:" + sys.prefix)
print("System version: "+sys.version)

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)


fname = "inputData/pricedata_LMP.csv" # FULL
#fname = "inputData/pricedata_LMP_100.csv" # Only 100 nodes
#fname = "inputData/pricedata_LMP_5.csv" # Only 5 nodes

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


def computePriceSweep(thisSlice):
    ###### BEGINNING OF MULTIPROCESSING FUNCTION ###########
    myLength = thisSlice.shape[1]

    # Simulation parameters - set these!
    storagePriceSet = np.arange(1.0,20.1,0.25)
#    storagePriceSet = np.arange(1.0,3.1,1)
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
    # print("Finished with problem setup")
    # sys.stdout.flush()

    # everythingStarts = time.time()
    # startTime = time.time()

    for myNodeName in thisSlice.index:
        ## Set up prices
    #     myNodeName = thisSlice.index[i]
        energyPrice = thisSlice.loc[myNodeName,:] / 1000.0 # Price $/kWh as array

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

    #     if ((i+1) % reportFrequency == 0): # Save our progress along the way
    #         elapsedTime = time.time()-startTime
    #         print("Finished node %s; \t%s computations in \t%.4f s \t(%.4f s/solve)" 
    #               % (i, scenariosPerReport,elapsedTime,elapsedTime/scenariosPerReport))
    #         sys.stdout.flush()
    #         sizeDf.to_csv('Data/VaryingPrices_StorageSizing_v2.csv')
    #         profitDf.to_csv('Data/VaryingPrices_StorageProfits_v2.csv')
    #         cycleDf.to_csv('Data/VaryingPrices_StorageCycles_v2.csv')

    # print("Done in %.3f s"%(time.time()-everythingStarts))
    return results

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
stopNode  = 0 # if set to zero, then will loop through all nodes
if ((stopNode == 0)|(stopNode > APNode_Prices.shape[0])): stopNode = APNode_Prices.shape[0]
    
APNode_Prices = APNode_Prices.ix[startNode:stopNode,startDate:endDate]
# someResults = computePriceSweep(thisSlice)


import multiprocessing

# Split dataset into roughly even chunks
j = min(multiprocessing.cpu_count(),10)
chunksize = (APNode_Prices.shape[0]/j)+1
splitFrames = [df for g,df in APNode_Prices.groupby(np.arange(APNode_Prices.shape[0])//chunksize)]
# chunksize = (thisSlice.shape[0]/j)+1
# splitFrames = [df for g,df in thisSlice.groupby(np.arange(thisSlice.shape[0])//chunksize)]

print("Entering the pool... bye-bye!")
sys.stdout.flush()
pool = multiprocessing.Pool(processes = j)
resultList = pool.map(computePriceSweep,splitFrames)
joinedResults = pd.concat(resultList)

joinedResults.sort_index(inplace=True)
sizeDf   = joinedResults.loc[(slice(None),'size'),:].reset_index(level=1,drop=True)
profitDf = joinedResults.loc[(slice(None),'profit'),:].reset_index(level=1,drop=True)
cycleDf  = joinedResults.loc[(slice(None),'kwhPassed'),:].reset_index(level=1,drop=True)

sizeDf.to_csv('Data/VaryingPrices_StorageSizing_v2.csv')
profitDf.to_csv('Data/VaryingPrices_StorageProfits_v2.csv')
cycleDf.to_csv('Data/VaryingPrices_StorageCycles_v2.csv')
print("All done!")
