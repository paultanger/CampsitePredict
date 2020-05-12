from flask import Flask, render_template
from bokeh.embed import components 
from datetime import datetime

import pandas as pd
import geopandas as gpd
import json
from bokeh.models import GeoJSONDataSource 
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, GMapOptions
from bokeh.plotting import gmap
from bokeh.models import BoxSelectTool
from bokeh.embed import components

app = Flask(__name__)

# https://pjandir.github.io/Bokeh-Heroku-Tutorial/
    
# the main (so like index)
@app.route('/')
def index():
    return render_template('index.html', title='main')

# subpage
@app.route('/home')
def home():
    
    # Read the Shapefile into GeoDataFrame
    # Calculate the x and y coordinates of the geometries into separate columns
    # Convert the GeoDataFrame into a Bokeh DataSource
    # Plot the x and y coordinates as points, lines or polygons (which are in Bokeh words: circle, multi_line and patches)
    
    # maybe this stuff doesn't need to be here?
    # for testing
    #output_file("gmap.html")
    # also try satellite or road
    map_options = GMapOptions(lat=30.2861, lng=-97.7394, map_type="hybrid", zoom=11)
    
    p = gmap("AIzaSyB9IAkbG2YcspA3G1PfxWl5CcmLfSEyr9Q", map_options, \
             title="iOverlander data", tools=['hover', 'pan', 'wheel_zoom'], \
                 toolbar_location="below")

    source = ColumnDataSource(
        data=dict(lat=[ 30.29,  30.20,  30.29],
                  lon=[-97.70, -97.74, -97.78])
    )
    
    p.circle(x="lon", y="lat", size=15, fill_color="blue", fill_alpha=0.8, source=source)
    
    # for testing
    #show(p)
    
    # for flask / heroku
    #curdoc().add_root(column(p))
    
    # these will be pasted into html
    
    script, div = components(p)

    # basic test
    #return render_template('home.html', title='home of maps')
    
    return render_template('home.html', script=script, div=div)

if __name__ == '__main__':
    # for testing
    #app.run(debug=True, use_reloader=True)
    app.run(port=33507)