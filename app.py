from flask import Flask, render_template
from bokeh.embed import components 
from datetime import datetime
app = Flask(__name__)
    
# the main (so like index)
@app.route('/')
def index():
    return render_template('index.html', title='main')

# subpage
@app.route('/home')
def home():
    
    # plot = figure(tools=TOOLS,
    #           title='Data from Quandle WIKI set',
    #           x_axis_label='date',
    #           x_axis_type='datetime')
    
    # script, div = components(plot)
    return render_template('home.html', title='home of maps')
    #return render_template('home.html', script=script, div=div)

if __name__ == '__main__':
    #app.run(debug=True, use_reloader=True)
    app.run(port=33507)