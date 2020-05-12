# from earth engine - need API

import ee
import datetime

ee.Authenticate()
ee.Initialize()

# test that things work
# Print the elevation of Mount Everest.
dem = ee.Image('USGS/SRTMGL1_003')
xy = ee.Geometry.Point([86.9250, 27.9881])
elev = dem.sample(xy, 30).first().get('elevation').getInfo()
print('Mount Everest elevation (m):', elev)

# TODO using the lat long from ioverlander, get the sat tiles around them.

# https://towardsdatascience.com/satellite-imagery-access-and-analysis-in-python-jupyter-notebooks-387971ece84b
from sentinelsat import SentinelAPI

user = 'detroitstylz' 
password = 'mysentinallogin' 
api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')
