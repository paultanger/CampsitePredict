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

app = Flask(__name__)



# https://pjandir.github.io/Bokeh-Heroku-Tutorial/

# Read the country borders shapefile into python using Geopandas 
# shapefile = 'data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp'
# gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]

# # Rename the columns
# gdf.columns = ['country', 'country_code', 'geometry']
# # Convert the GeoDataFrame to GeoJSON format so it can be read by Bokeh
# merged_json = json.loads(gdf.to_json())
# json_data = json.dumps(merged_json)
# geosource = GeoJSONDataSource(geojson=json_data)

# # Make the plot
# TOOLTIPS = [
# ('UN country', '@country')
# ]

# p = figure(title='World Map', plot_height=600 , plot_width=950, tooltips=TOOLTIPS,
# x_axis_label='Longitude', y_axis_label='Latitude')

# p.patches('xs','ys', source=geosource, fill_color='white', line_color='black',
# hover_fill_color='lightblue', hover_line_color='black')
 
# # This final command is required to launch the plot in the browser
# curdoc().add_root(column(p))
    
# the main (so like index)
@app.route('/')
def index():
    return render_template('index.html', title='main')

# subpage
@app.route('/home')
def home():
    
    output_file("gmap.html")

    map_options = GMapOptions(lat=30.2861, lng=-97.7394, map_type="roadmap", zoom=11)
    
    p = gmap("AIzaSyB9IAkbG2YcspA3G1PfxWl5CcmLfSEyr9Q", map_options, title="Austin")
    
    source = ColumnDataSource(
        data=dict(lat=[ 30.29,  30.20,  30.29],
                  lon=[-97.70, -97.74, -97.78])
    )
    
    p.circle(x="lon", y="lat", size=15, fill_color="blue", fill_alpha=0.8, source=source)
    
    # for testing
    #show(p)
    
    # for flask / heroku
    curdoc().add_root(column(p))

    # plot = figure(tools=TOOLS,
    #           title='Data from Quandle WIKI set',
    #           x_axis_label='date',
    #           x_axis_type='datetime')
    
    # script, div = components(plot)
    
    #return render_template('home.html', title='home of maps')
    
    #return render_template('home.html', script=script, div=div)

if __name__ == '__main__':
    #app.run(debug=True, use_reloader=True)
    app.run(port=33507)