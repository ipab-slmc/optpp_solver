#!/usr/bin/env python

import pyexotica as exo
from numpy import array
from numpy import matrix
import math
from pyexotica.publish_trajectory import *
from time import sleep


def figure_eight(t):
    return array([0.6, -0.1 + math.sin(t * 2.0 * math.pi * 0.5) * 0.1, 0.5 + math.sin(t * math.pi * 0.5) * 0.2, 0, 0, 0])

exo.Setup.init_ros()
(sol, prob) = exo.Initializers.load_xml_full(
    '{optpp_solver}/resources/optpp_ik.xml')
problem = exo.Setup.create_problem(prob)
solver = exo.Setup.create_solver(sol)
solver.specify_problem(problem)

import signal


def sig_int_handler(signal, frame):
    raise KeyboardInterrupt
signal.signal(signal.SIGINT, sig_int_handler)

dt = 0.002
t = 0.0
q = array([0.0]*7)
print('Publishing IK')
while True:
    try:
        problem.set_goal('Position', figure_eight(t))
        problem.start_state = q
        q = solver.solve()[0]
        publish_pose(q, problem)
        sleep(dt)
        t = t+dt
    except KeyboardInterrupt:
        break

# Avoid heap issues
del problem, solver
