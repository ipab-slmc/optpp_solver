#!/usr/bin/env python

import pyexotica as exo
from pyexotica.publish_trajectory import *

exo.Setup.init_ros()
solver=exo.Setup.loadSolver('{optpp_solver}/resources/optpp_traj.xml')
problem = solver.getProblem()

for t in range(0,problem.T):
    if float(t)*problem.tau<0.8:
        problem.set_rho('Frame',0.0,t)
    else:
        problem.set_rho('Frame',1e5,t)

solution = solver.solve()

plot(solution)

publishTrajectory(solution, problem.T*problem.tau, problem)

