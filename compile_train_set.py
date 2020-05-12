import pandas as pd
import numpy as np

# set wd
wd = 'data/'

# TODO implement web scrape for latest data from here: http://app.ioverlander.com/countries/places_by_country

# get data
USdata = pd.read_csv(wd + 'iOverlander Places - United States - 2020-04-29.csv')

# subset just some categories
USdata_filtered = USdata[USdata.Category.isin(['Wild Camping', 'Informal Campsite'])]

# keep cols we need
USdata_filtered = USdata_filtered.iloc[:,0:6]

# convert coordinates to x and y for bokeh mapping
import math
from ast import literal_eval

# function to do this
def merc(coords):
    lat = coords[0]
    long = coords[1]
    r_major = 6378137.000
    
    x = r_major * math.radians(long)
    scale = x/long
    
    y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 + lat * (math.pi/180.0)/2.0)) * scale
    #print(lat, long)
    return (x, y)

# USdata_filtered.at[0,'Latitude']

# TODO: clean up and optimize this part
x, y = merc(USdata_filtered.at[0,'Latitude'], USdata_filtered.at[0,'Longitude'])

USdata_filtered['coords'] = list(zip(USdata_filtered.Latitude, USdata_filtered.Longitude))

USdata_filteredxy = USdata_filtered.copy()

USdata_filteredxy['xy_coords'] = USdata_filtered['coords'].apply(lambda x: merc(x))
USdata_filteredxy['x'] = USdata_filteredxy['xy_coords']

USdata_filteredxy['x'], USdata_filteredxy['y'] = zip(*USdata_filteredxy.xy_coords)

# save data for app
USdata_filteredxy.to_csv('data/USdata_filteredxy.csv')


# convert to geopandas and maybe a shapefile
import geopandas

USsites_gdf = geopandas.GeoDataFrame(
    USdata_filtered, geometry=geopandas.points_from_xy(USdata_filtered.Longitude, USdata_filtered.Latitude))

# save to shapefile or geoJSON:
USsites_gdf.to_file("countries.shp")
USsites_gdf.to_file("countries.geojson", driver='GeoJSON')
