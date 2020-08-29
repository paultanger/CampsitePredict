#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:43:04 2020
@author: paultanger
"""

from flask import Flask, render_template
from bokeh.embed import components 
from bokeh.models import HoverTool
from bokeh.models import GeoJSONDataSource 
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column

app = Flask(__name__)

# the main (so like index)
@app.route('/')
def index():
    return render_template('index.html', title='main')

# subpage
@app.route('/home')
def home():
    return render_template('home.html', title='home of maps')

if __name__ == "__main__":
    app.run(debug=True)
    #app.run(port=33507)
