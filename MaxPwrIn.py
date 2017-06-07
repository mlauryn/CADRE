""" Optimization of the CADRE roll angle, fin angle and set point current for maximum solar power input"""

import time
t = time.time()

from six.moves import range

import numpy as np

from openmdao.api import IndepVarComp, Component, Group, Problem, ScipyOptimizer
#from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
 
from CADRE.power import Power_SolarPower, Power_CellVoltage
from CADRE.parameters import BsplineParameters
from CADRE.attitude import Attitude_Angular, Attitude_AngularRates, Attitude_Attitude, \
     Attitude_Roll, Attitude_RotationMtx, \
     Attitude_RotationMtxRates, Attitude_Sideslip, Attitude_Torque
from CADRE.orbit import Orbit_Initial, Orbit_Dynamics
from CADRE.solar import Solar_ExposedArea
from CADRE.sun import Sun_LOS, Sun_PositionBody, Sun_PositionECI, Sun_PositionSpherical
from CADRE.power import Power_CellVoltage, Power_SolarPower, Power_Total

import os
import pickle

class CADRE(Group):
    """ OpenMDAO implementation of the CADRE model.

    Args
    ----
    n : int
        Number of time integration points.

    m : int
        Number of panels in fin discretization.

    solar_raw1 : string
        Name of file containing the angle map of solar exposed area.

    solar_raw2 : string
        Name of file containing the azimuth map of solar exposed area.

    comm_raw : string
        Name of file containing the az-el map of transmitter gain.

    power_raw : string
        Name of file containing a map of solar cell output voltage versus current,
        area, and temperature.

    initial_params : dict, optional
        Dictionary of initial values for all parameters. Set the keys you want to
        change.
    """

    def __init__(self, n, m, solar_raw1=None, solar_raw2=None, comm_raw=None,
                 power_raw=None, initial_params=None):

        super(CADRE, self).__init__()

        self.ln_solver.options['mode'] = 'auto'

        # Analysis parameters
        self.n = n
        self.m = m
        h =5400.0 / (self.n - 1)

        # User-defined initial parameters
        if initial_params is None:
            initial_params = {}
        if 't1' not in initial_params:
            initial_params['t1'] = 0.0
        if 't2' not in initial_params:
            initial_params['t2'] =5400.0
        if 't' not in initial_params:
            initial_params['t'] = np.array(range(0, n))*h
        if 'CP_Isetpt' not in initial_params:
            initial_params['CP_Isetpt'] = 0.2 * np.ones((12, self.m))
        if 'CP_gamma' not in initial_params:
            initial_params['CP_gamma'] = np.pi/4 * np.ones((self.m, ))
        if 'CP_P_comm' not in initial_params:
            initial_params['CP_P_comm'] = 0.1 * np.ones((self.m, ))

        if 'iSOC' not in initial_params:
            initial_params['iSOC'] = np.array([0.5])

        # Fixed Station Parameters for the CADRE problem.
        # McMurdo station: -77.85, 166.666667
        # Ann Arbor: 42.2708, -83.7264
        if 'LD' not in initial_params:
            initial_params['LD'] = 5000.0
        if 'lat' not in initial_params:
            initial_params['lat'] = 42.2708
        if 'lon' not in initial_params:
            initial_params['lon'] = -83.7264
        if 'alt' not in initial_params:
            initial_params['alt'] = 0.256

        # Initial Orbital Elements
        if 'r_e2b_I0' not in initial_params:
            initial_params['r_e2b_I0'] = np.zeros((6, ))

        # Some initial setup.
        self.add('p_t1', IndepVarComp('t1', initial_params['t1']), promotes=['*'])
        self.add('p_t2', IndepVarComp('t2', initial_params['t2']), promotes=['*'])
        self.add('p_t', IndepVarComp('t', initial_params['t']), promotes=['*'])

        # Design parameters
        self.add('p_CP_Isetpt', IndepVarComp('CP_Isetpt',
                                           initial_params['CP_Isetpt']),
                  promotes=['*'])
        self.add('p_CP_gamma', IndepVarComp('CP_gamma',
                                         initial_params['CP_gamma']),
                  promotes=['*'])
        # self.add('p_CP_P_comm', IndepVarComp('CP_P_comm',
                                          # initial_params['CP_P_comm']),
                 # promotes=['*'])
        # self.add('p_iSOC', IndepVarComp('iSOC', initial_params['iSOC']),
                 # promotes=['*'])

        # These are broadcast params in the MDP.
        #self.add('p_cellInstd', IndepVarComp('cellInstd', np.ones((7, 12))),
        #         promotes=['*'])
        self.add('p_finAngle', IndepVarComp('finAngle', np.pi / 9), promotes=['*'])
        #self.add('p_antAngle', IndepVarComp('antAngle', 0.0), promotes=['*'])

        self.add('param_LD', IndepVarComp('LD', initial_params['LD']),
                 promotes=['*'])
        #self.add('param_lat', IndepVarComp('lat', initial_params['lat']),
        #         promotes=['*'])
        #self.add('param_lon', IndepVarComp('lon', initial_params['lon']),
        #         promotes=['*'])
        #self.add('param_alt', IndepVarComp('alt', initial_params['alt']),
        #         promotes=['*'])
        self.add('param_r_e2b_I0', IndepVarComp('r_e2b_I0',
                                             initial_params['r_e2b_I0']),
                 promotes=['*'])

        # Add Component Models
        self.add("BsplineParameters", BsplineParameters(n, m), promotes=['*'])
        self.add("Attitude_Angular", Attitude_Angular(n), promotes=['*'])
        self.add("Attitude_AngularRates", Attitude_AngularRates(n, h), promotes=['*'])
        self.add("Attitude_Attitude", Attitude_Attitude(n), promotes=['*'])
        self.add("Attitude_Roll", Attitude_Roll(n), promotes=['*'])
        self.add("Attitude_RotationMtx", Attitude_RotationMtx(n), promotes=['*'])
        self.add("Attitude_RotationMtxRates", Attitude_RotationMtxRates(n, h),
                 promotes=['*'])

        # Not needed?
        #self.add("Attitude_Sideslip", Attitude_Sideslip(n))

        #self.add("Attitude_Torque", Attitude_Torque(n), promotes=['*'])
        # self.add("BatteryConstraints", BatteryConstraints(n), promotes=['*'])
        # self.add("BatteryPower", BatteryPower(n), promotes=['*'])
        # self.add("BatterySOC", BatterySOC(n, h), promotes=['*'])
        # self.add("Comm_AntRotation", Comm_AntRotation(n), promotes=['*'])
        # self.add("Comm_AntRotationMtx", Comm_AntRotationMtx(n), promotes=['*'])
        # self.add("Comm_BitRate", Comm_BitRate(n), promotes=['*'])
        # self.add("Comm_DataDownloaded", Comm_DataDownloaded(n, h), promotes=['*'])
        # self.add("Comm_Distance", Comm_Distance(n), promotes=['*'])
        # self.add("Comm_EarthsSpin", Comm_EarthsSpin(n), promotes=['*'])
        # self.add("Comm_EarthsSpinMtx", Comm_EarthsSpinMtx(n), promotes=['*'])
        # self.add("Comm_GainPattern", Comm_GainPattern(n, comm_raw), promotes=['*'])
        # self.add("Comm_GSposEarth", Comm_GSposEarth(n), promotes=['*'])
        # self.add("Comm_GSposECI", Comm_GSposECI(n), promotes=['*'])
        # self.add("Comm_LOS", Comm_LOS(n), promotes=['*'])
        # self.add("Comm_VectorAnt", Comm_VectorAnt(n), promotes=['*'])
        # self.add("Comm_VectorBody", Comm_VectorBody(n), promotes=['*'])
        # self.add("Comm_VectorECI", Comm_VectorECI(n), promotes=['*'])
        # self.add("Comm_VectorSpherical", Comm_VectorSpherical(n), promotes=['*'])

        # Not needed?
        #self.add("Orbit_Initial", Orbit_Initial(), promotes=['*'])

        self.add("Orbit_Dynamics", Orbit_Dynamics(n, h), promotes=['*'])
        self.add("Voltage", Power_CellVoltage(n, power_raw),
                 promotes=['*'])
        self.add("Power", Power_SolarPower(n), promotes=['*'])
        #self.add("Power_Total", Power_Total(n), promotes=['*'])

        # Not needed?
        #self.add("ReactionWheel_Motor", ReactionWheel_Motor(n))

        # self.add("ReactionWheel_Power", ReactionWheel_Power(n), promotes=['*'])
        # self.add("ReactionWheel_Torque", ReactionWheel_Torque(n), promotes=['*'])
        # self.add("ReactionWheel_Dynamics", ReactionWheel_Dynamics(n, h), promotes=['*'])
        self.add("Solar_ExposedArea", Solar_ExposedArea(n, solar_raw1,
                                                        solar_raw2), promotes=['*'])
        self.add("Sun_LOS", Sun_LOS(n), promotes=['*'])
        self.add("Sun_PositionBody", Sun_PositionBody(n), promotes=['*'])
        self.add("Sun_PositionECI", Sun_PositionECI(n), promotes=['*'])
        self.add("Sun_PositionSpherical", Sun_PositionSpherical(n), promotes=['*'])
        # self.add("ThermalTemperature", ThermalTemperature(n, h), promotes=['*'])

class PowerSum(Component):
  """ Definition of objective function."""
  def __init__(self, n):
    super(PowerSum, self).__init__()
    self.add_param('P_sol', np.zeros((n, )), units="W",
                        desc="Solar panels power over time")
    self.add_output("result", 0.0)

    self.J = -np.ones((1, n))

  def solve_nonlinear(self, params, unknowns, resids):
    unknowns['result'] = -np.sum(params['P_sol'])

  def linearize(self, params, unknowns, resids):

    return {("result", "P_sol") : self.J }


class MaxPwrIn(Group):
  """Setup of initial parameters for the problem"""
  def __init__(self):
    super(MaxPwrIn, self).__init__()
    n = 90 
    m = 30
    #LS2_orbit parameters 
    initial_params = {'CP_Isetpt':np.loadtxt('CP_Isetpt.out'), 'LD':6383.167, 'r_e2b_I0':np.array([-3994.154, -5599.597, 0., -0.8089412, 0.57701217, 7.547477])} 
    
    self.add("CADRE", CADRE(n, m, initial_params=initial_params))
    self.add("PowerSum", PowerSum(n))

    self.connect("CADRE.P_sol", "PowerSum.P_sol")


if __name__ == "__main__":

  model = Problem()
  model.root = MaxPwrIn()
  
  # Run model to get baseline average orbit power value
  model.setup()
  model.run()
   
  Pawg1 = -model['PowerSum.result']/89
  
  import matplotlib
  matplotlib.use('Agg')  
  import pylab

  pylab.figure()
  pylab.title("Roll angle $\gamma$, Before optimization")
  pylab.ylabel('$\gama$, degrees')
  pylab.xlabel('time,s')
  pylab.subplot(211)
  pylab.plot(model['CADRE.CP_gamma'] * 180/np.pi)

  #add driver
  model.driver = ScipyOptimizer()
  model.driver.options['optimizer'] = "SLSQP"
  #model.driver.options['tol'] = 1.0e-8 
  
  model.driver.add_desvar("CADRE.CP_gamma", lower=0, upper=np.pi/2.)
  #model.driver.add_desvar("CADRE.CP_Isetpt", lower=0., upper=0.4)
  model.driver.add_desvar("CADRE.finAngle", lower=0., upper=np.pi/2)
  model.driver.add_objective("PowerSum.result")
  
  model.setup()
  model.run()
  
  #output design variable results
  #np.savetxt('CP_Isetpt.out', model['CADRE.CP_Isetpt'], fmt='%1.4e')

  pylab.title("After Optimization")
  pylab.subplot(212)
  pylab.plot(model['CADRE.CP_gamma'] * 180/np.pi)
  pylab.savefig('Gamma.png')
 
  Pawg2 = -model['PowerSum.result']/89
  
  print("Orbit average power before optimization: %.3f W" %Pawg1)
  print("Orbit average power after optimization: %.3f W" %Pawg2)

  angle = model['CADRE.finAngle']*180/np.pi
  print("Optimal fin angle: %.2f degrees" %angle)

  print('run time: ', time.time() - t)
  #pylab.show()
