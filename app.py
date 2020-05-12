from flask import Flask, render_template
from datetime import datetime
app = Flask(__name__)
    
# the main (so like index)
@app.route('/')
def index():
    return render_template('index.html', title='main')

# subpage
@app.route('/home')
def home():
    return render_template('home.html', title='home of maps')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)