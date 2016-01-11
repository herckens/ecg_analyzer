import numpy
from matplotlib import pylab as pl

import database

db = database.DataBase()
data = db.get_data()

# Plot first measurement available
pl.ion()
pl.plot(data[0,:])
#pl.plot(data[1,:])
pl.show()
