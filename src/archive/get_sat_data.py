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

wd = 'data/test_shp/'

# use shp to define which tile
landsat = 
geometry = 

landsat = ee.Image('LANDSAT/LC08/C01/T1_TOA/LC08_123032_20140515')
  .select(['B4', 'B3', 'B2']);
  
geometry = ee.Geometry.Rectangle([116.2621, 39.8412, 116.4849, 40.01236]);


test = ee.batch.Export.image.toDrive(landsat, 'imageToDriveExample', scale=30, region=geometry)
test.start()
# Error: Exported bands must have compatible data types; found inconsistent types: Float32 and UInt16.



# https://towardsdatascience.com/satellite-imagery-access-and-analysis-in-python-jupyter-notebooks-387971ece84b
from sentinelsat import SentinelAPI
import getpass
user = 'detroitstylz' 
password = getpass.getpass()
api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')
