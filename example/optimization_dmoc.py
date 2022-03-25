import sys
import os
#!/usr/bin/env python
# coding=UTF-8
'''
Author: Wei Luo
Date: 2022-03-15 13:58:38
LastEditors: Wei Luo
LastEditTime: 2022-03-25 13:06:33
Note: Note
'''

BASEPATH = os.path.abspath(__file__).split('rpg_time_optimal',
                                           1)[0] + 'rpg_time_optimal/'
sys.path += [BASEPATH + 'src']
from track import Track
from quad_dmoc import Quad
# from integrator import RungeKutta4
from planner_itm_dmoc import Planner
from trajectory_itm_dmoc import Trajectory
from plot_itm_dmoc import CallbackPlot

track = Track(BASEPATH + "/tracks/track.yaml")
quad = Quad(BASEPATH + "/quads/quad.yaml")

cp = CallbackPlot(pos='xy',
                  vel='xya',
                  ori='xyzw',
                  rate='xyz',
                  inputs='u',
                  prog='mn')
planner = Planner(quad, track, {
    'tolerance': 0.3,
    'nodes_per_gate': 40,
    'vel_guess': 3.0
})
planner.setup()
planner.set_iteration_callback(cp)
x = planner.solve()

traj = Trajectory(x, NPW=planner.NPW, wp=planner.wp)
traj.save(BASEPATH + '/example/result_cpc_format_itm_dmoc.csv', False)
traj.save(BASEPATH + '/example/result_itm_dmoc.csv', True)
