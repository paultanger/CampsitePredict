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
import boto3

app = Flask(__name__)

# the main (so like index)
@app.route('/')
def index():
    return render_template('index.html', title='main')

# subpage
@app.route('/home')
def home():
    
    #USdata_filteredxy = pd.read_csv('USdata_filteredxy.csv')
    USdata_filteredxy = pd.read_csv('s3://campsiteprediction/heroku_data/USdata_filteredxy.csv')

    # center over ridgway, CO
    map_options = GMapOptions(lat=38.1584, lng=-107.7697, map_type="hybrid", zoom=5)
    
    s3 = boto3.resource('s3')
    obj = s3.Object('campsiteprediction', 'heroku_data/google_API_key')
    API_key = obj.get()['Body'].read().decode('utf-8').strip()
    p = gmap(API_key, map_options, \
              tools=['hover', 'pan', 'wheel_zoom'], \
                  toolbar_location="below") 
        
    p.plot_height=600
    p.plot_width=1000

    # google maps uses regular GPS not x y!
    p.circle(x= USdata_filteredxy['Longitude'], y= USdata_filteredxy['Latitude'], size=6, fill_color="blue", fill_alpha=0.8)

    # these will be pasted into html
    
    script, div = components(p)

    # basic test
    #return render_template('home.html', title='home of maps')
    return render_template('home.html', script=script, div=div)

# @app.errorhandler(404)
# def page_not_found(error):
#     """Custom 404 page."""
#     return render_template('404.html'), 404

if __name__ == '__main__':
    # for testing
    #app.run(debug=True, use_reloader=True)
    app.run(port=33507)