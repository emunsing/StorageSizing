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
import sys, os, pickle


from simulationFunctions import *

import sys  # This is necessary for printing updates within a code block, via sys.stdout.flush()
import time # Use time.sleep(secs) to sleep a process if needed

print("Running Storage Efficiency sweep")
print("Current environment directory:" + sys.prefix)
print("System version: "+sys.version)

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)


try:
    fname = os.environ['INPUTFILE']
except KeyError:
    fname = "inputData/pricedata_LMP_100.csv" # Only 100 nodes    

print("Running Storage Efficiency sweep with input file "+fname)
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

def efficiencySweep(thisSlice):
    # Simulation parameters
    # myEfficiencies = [0.6,0.8,0.9]

    pid = multiprocessing.current_process().pid
    myEfficiencies = np.arange(0.4,1.01,0.02)
    reservoirSize=1
    E_min = 0
    E_max = 1
    # The 1-hour system will be SOC constrained, rather than power constrained. We accordingly don't worry about P_max and P_min

    lastEfficiency = 0  # This is used to track whether the efficiency has switched
    storagePrice = 0 * simulationYears # Amortized cost of storage
    myLength = thisSlice.shape[1]


    # Result dataframe: Size, kwhPassed, and profits for each node, at each efficiency (columns)
    resultIndex = pd.MultiIndex.from_product([thisSlice.index,['cycleCount','storageProfit']])
    results = pd.DataFrame(index = resultIndex, columns=myEfficiencies)
    powerOut = pd.DataFrame(index = thisSlice.index, columns = thisSlice.columns)

    # Build basic model, with everything except the state transition constraints
    # For each node,
    #  Set the cost function to be the prices for that period

    # For each efficiency,
    #  if the new efficiency is not the old efficiency:
    #    Add the state transition constraints with name 'chargeCons'
    #  Run the simulation
    #  Remove the state transition constraint

    model = CyClpSimplex()
    model.logLevel = 0
    x_var = model.addVariable('x',myLength*3+2)
    h_constant = sps.hstack( [1, sps.coo_matrix((1, myLength*3+1))] ) # Force h to be a specific size:         
    (A,b) = createABineq_noPowerConstraint(myLength, E_min, E_max)
    model.addConstraint(h_constant * x_var == reservoirSize,'fixedSize')
    model.addConstraint(         A * x_var <= b.toarray(),  'inequalities')


    #### LOOP THROUGH nodes
    for i in range(thisSlice.shape[0]):
        # Define cost function
        myNodeName = thisSlice.index[i]
        energyPrice = thisSlice.loc[myNodeName,:] / 1000.0 # Price $/kWh as array
        c = np.concatenate([[storagePrice],[0]*(myLength+1),energyPrice,energyPrice],axis=0)  # No cost for storage state; charged for what we consume, get paid for what we discharge
        c_clp = CyLPArray(c)
        model.objective = c_clp * x_var

        for eff_round in myEfficiencies:

            if eff_round != lastEfficiency:  # If we just switched nodes (and not efficiencies) don't bother updating efficiencies
                try:
                    model.removeConstraint('equalities')
                except:
                    pass
                (eff_in, eff_out) = [np.sqrt(eff_round)] *2  # Properly account for the round trip efficiency of storage
                (A_eq, b_eq) = createABeq(myLength, delta_T, eff_in, eff_out)
                model.addConstraint(A_eq * x_var == b_eq.toarray(),'equalities')

            model.primal()  # Solve

            x = model.primalVariableSolution['x']
            results.loc[(myNodeName,'storageProfit'),eff_round] = np.dot(-c, x)   #Calculate profits at optimal storage level
            c_grid = x[2+myLength : 2+myLength*2]
            results.loc[(myNodeName,'cycleCount'),   eff_round] = sum(c_grid)*eff_in # net kWh traveled

            if round(eff_round,2) == 0.9:
                powerOut.loc[myNodeName,:] = x[2+myLength : 2+myLength*2] + x[2+myLength*2 : 2+myLength*3]

            lastEfficiency = eff_round

        # Done with the loop; reverse the efficiency set and move on to the next node
        myEfficiencies = myEfficiencies[::-1]

        if (i+1)%50==0:
        	results.to_csv( 'Data/efficiencyResults_pid'+str(pid)+'temp.csv' )
        	powerOut.to_csv('Data/efficiencyPower_pid'  +str(pid)+'temp.csv' )

    return (results,powerOut,pid)

## CUSTOM START/END DATE
startDate = parser.parse('01/01/12 00:00')  # year starts at 2013-01-01 00:00:00

try:
    endDate=parser.parse(os.environ['ENDDATE'])
except KeyError:
    endDate=parser.parse('01/31/12 23:00') # year ends at 2013-12-31 23:00:00

startDate = pytz.timezone('America/Los_Angeles').localize(startDate).astimezone(pytz.utc)
endDate = pytz.timezone('America/Los_Angeles').localize(endDate).astimezone(pytz.utc)

# ## FULL DATASET
# startDate = APNode_Prices.columns.values[ 0].astype('M8[m]').astype('O') # Convert to datetime, not timestamp
# endDate   = APNode_Prices.columns.values[-1].astype('M8[m]').astype('O')
# startDate = pytz.utc.localize(startDate)
# endDate   = pytz.utc.localize(endDate)

timespan = relativedelta.relativedelta(endDate +timestep, startDate)
simulationYears = timespan.years + timespan.months/12. + timespan.days/365. + timespan.hours/8760.  # Leap years will be slightly more than a year, and that's ok.

startNode = 0

try:
    stopNode = int(os.environ['STOPNODE'])
except KeyError:
    stopNode  = 15 # if set to zero, then will loop through all nodes

if ((stopNode == 0)|(stopNode > APNode_Prices.shape[0])): stopNode = APNode_Prices.shape[0]
thisSlice = APNode_Prices.ix[startNode:stopNode,startDate:endDate]

nodeList = thisSlice.index.values

try:  # If we've saved a pickled file of the nodes that we want to hang onto
    nodesFromFile = os.environ['NODELIST']
    if nodesFromFile:
        with open('nodeList.pkl','rb')as f:
            nodeList = pickle.loads(f.read())
except (KeyError, IOError):
    pass

thisSlice = APNode_Prices.ix[nodeList,startDate:endDate]
print("Working with a slice of data with %s nodes from %s to %s"%(thisSlice.shape[0],thisSlice.columns.values[0],thisSlice.columns.values[-1]))

import multiprocessing

# Split dataset into roughly even chunks
j = min(multiprocessing.cpu_count(),10)
# chunksize = (APNode_Prices.shape[0]/j)+1
# splitFrames = [df for g,df in APNode_Prices.groupby(np.arange(APNode_Prices.shape[0])//chunksize)]
chunksize = (thisSlice.shape[0]/j)+1
splitFrames = [df for g,df in thisSlice.groupby(np.arange(thisSlice.shape[0])//chunksize)]

print("Entering the pool... bye-bye!")
solverStartTime = time.time()

pool = multiprocessing.Pool(processes = j)
resultList = pool.map(efficiencySweep,splitFrames) # Each worker returns a tuple of (result,PowerOut,pid)

(resultFrames, powerOutputs, pids) = zip(*resultList)

results = pd.concat(resultFrames).sort_index()
powerResults = pd.concat(powerOutputs).sort_index()

profitDf = results.loc[(slice(None),'storageProfit'),:].reset_index(level=1,drop=True)
cycleDf  = results.loc[(slice(None),'cycleCount'),:].reset_index(level=1,drop=True)

profitDf.to_csv('Data/kwhValue_step_02.csv')
cycleDf.to_csv('Data/cycleCount_step_02.csv')
powerResults.to_csv('Data/powerOutput_90pct.csv')

for pid in pids:
    try:
    	os.remove('Data/efficiencyResults_pid'+str(pid)+'temp.csv')
        os.remove('Data/efficiencyPower_pid'  +str(pid)+'temp.csv')
    except OSError:
        pass  # We probably didn't have enough datapoints to make this relevant

print('Total function call time: %.3f seconds' % (time.time() - solverStartTime))
