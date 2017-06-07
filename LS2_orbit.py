import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

#epoch

#Septermber equinox in year 2017
date = ['2017-06-23 16:00']
time = Time(date, format='iso', scale='utc')
epoch = time.jyear_str

#orbit parameters
altitude = 500.0 * u.km
inclination = 97.5 * u.deg
RAAN = 234.5 * u.deg

orbit = Orbit.circular(Earth, altitude, inclination, RAAN, epoch=epoch)
r = Orbit.rv(orbit)

print(r) 
 


