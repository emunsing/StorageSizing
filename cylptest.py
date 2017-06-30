from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
import numpy as np

model = CyClpSimplex()

x_var = model.addVariable('x',3)

A = np.matrix([[1.,2.,0],[1.,0,1.]])
c = np.array([1.,2.,3.])
c_clp = CyLPArray([1.,2.,3.])

print(x_var.value)
print(c_clp.value)

model.objective = c_clp * x_var