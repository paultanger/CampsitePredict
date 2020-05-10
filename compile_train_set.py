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