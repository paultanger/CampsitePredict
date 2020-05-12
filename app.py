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
import s3fs

app = Flask(__name__)

# https://pjandir.github.io/Bokeh-Heroku-Tutorial/
# https://towardsdatascience.com/exploring-and-visualizing-chicago-transit-data-using-pandas-and-bokeh-part-ii-intro-to-bokeh-5dca6c5ced10

# the main (so like index)
@app.route('/')
def index():
    return render_template('index.html', title='main')

# subpage
@app.route('/home')
def home():
    
    #USdata_filteredxy = pd.read_csv('USdata_filteredxy.csv')
    USdata_filteredxy = pd.read_csv('s3://campsiteprediction/heroku_data/USdata_filteredxy.csv')

    # Read the Shapefile into GeoDataFrame
    # Calculate the x and y coordinates of the geometries into separate columns
    # Convert the GeoDataFrame into a Bokeh DataSource
    # Plot the x and y coordinates as points, lines or polygons (which are in Bokeh words: circle, multi_line and patches)
    
    # maybe this stuff doesn't need to be here?
    # for testing
    #output_file("gmap.html")
    # also try satellite or road
    # center over ridgway, CO
    map_options = GMapOptions(lat=38.1584, lng=-107.7697, map_type="hybrid", zoom=5)
    
    # get API key
    with open('google_API_key') as f:
        API_key = f.read().strip()
    
    p = gmap(API_key, map_options, \
             title="iOverlander data", tools=['hover', 'pan', 'wheel_zoom'], \
                 toolbar_location="below") 
    # this doesn't work... 
    # TODO: make wheel scroll zoom active by default
    #p.toolbar.active_scroll = p.select_one('wheel_zoom')
    # source = ColumnDataSource(
    #     data=dict(lat=[ 30.29,  30.20,  30.29],
    #               lon=[-97.70, -97.74, -97.78])
    #)
    
    # google maps uses regluar GPS not x y!
    #p.circle(x= USdata_filteredxy['x'], y= USdata_filteredxy['y'], size=6, fill_color="blue", fill_alpha=0.8)
    #p.circle(x= 3932604.694, y= -12929412.32, size=6, fill_color="blue", fill_alpha=0.8)
    p.circle(x= USdata_filteredxy['Longitude'], y= USdata_filteredxy['Latitude'], size=6, fill_color="blue", fill_alpha=0.8)

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