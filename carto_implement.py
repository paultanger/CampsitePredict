#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:55:39 2020

@author: paultanger
"""

# https://carto.com/developers/cartoframes/guides/Quickstart/

from cartoframes.data.observatory import Enrichment
from cartoframes.data.services import Geocoding, Isolines
from cartoframes.viz import Map, color_continuous_style, size_continuous_style
import pandas as pd

from cartoframes.auth import Credentials
from cartoframes.auth import set_default_credentials

creds_file = 'carto_API_cred.json'
carto_creds = Credentials.from_file(creds_file)
set_default_credentials(creds_file)

stores_df = pd.read_csv('http://libs.cartocdn.com/cartoframes/files/starbucks_brooklyn.csv')


from cartoframes.data.services import Geocoding

stores_gdf, _ = Geocoding().geocode(stores_df, street='address')
stores_gdf.head()

from cartoframes.viz import Map, Layer

result_map = Map(Layer(stores_gdf))

result_map.publish('starbucks_analysis', password=None, if_exists='replace')

# https://paultanger.carto.com/kuviz/3fdc902c-c74c-4084-a2a4-f6fd2277e696