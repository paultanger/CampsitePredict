#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:43:04 2020

@author: paultanger
"""

from flask import Flask, render_template

from bokeh.embed import components 
from bokeh.models import HoverTool
from bokeh.charts import Scatter

import json

from bokeh.models import GeoJSONDataSource 
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column

import pandas as pd
import geopandas as gpd


app = Flask(__name__)

# the main (so like index)
@app.route('/')
def index():
    return render_template('index.html', title='main')

# subpage
@app.route('/home')
def home():
    return render_template('home.html', title='home of maps')

# https://medium.com/@jodorning/how-to-deploy-a-bokeh-app-on-heroku-486d7db28299

# Convert the GeoDataFrame to GeoJSON format so it can be read by Bokeh
merged_json = json.loads(gdf.to_json())
json_data = json.dumps(merged_json)
geosource = GeoJSONDataSource(geojson=json_data)

# Make the plot
TOOLTIPS = [
('UN country', '@country')
]

p = figure(title='World Map', plot_height=600 , plot_width=950, tooltips=TOOLTIPS,
x_axis_label='Longitude', y_axis_label='Latitude')

p.patches('xs','ys', source=geosource, fill_color='white', line_color='black',
hover_fill_color='lightblue', hover_line_color='black')
 
# This final command is required to launch the plot in the browser
curdoc().add_root(column(p))

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(port=33507)
