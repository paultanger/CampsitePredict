from flask import Flask, request, render_template, send_from_directory
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from getpass import getpass
import sys, os

# from flask_sqlalchemy import SQLAlchemy
# db = SQLAlchemy(app)

def setup_db(db_details):
    return create_engine(db_details)

# setup db for queries
# db_details = f'postgresql://postgres:{getpass()}@3.20.229.59:5432/campsite'
# engine = setup_db(db_details)

data = pd.read_csv('static/data/df_with_preds_no_imgs3.tsv', sep='\t')
# select cols to keep for display
data_display = data[['predict', 'actual', 'correct', 'filename', 'Name', 'Category', 'Description', 'State']]

# setup image path
img_path = '../../../media/'

# app = Flask(__name__)
app = Flask(__name__, root_path='./')# template_folder = 'templates/')
# app = Flask(__name__, root_path='./', static_url_path='/Users/pault/Desktop/github/media/', 
# app = Flask(__name__, root_path='./', static_url_path='/Users/pault/Desktop/github/media/') 


# MEDIA_FOLDER = '../../../media/images/'
MEDIA_FOLDER = '/Users/pault/Desktop/github/media/images/'
# MEDIA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
@app.route('/images/<path:filename>')
def get_file(filename):
    return send_from_directory(MEDIA_FOLDER, filename, as_attachment=True)

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

# display sample from df for testing
@app.route('/analysis') 
def analysis():
    sample = data_display.sample(10)
    return render_template("analysis.html", tables=[sample.to_html(classes='data', index=False)], titles=sample.columns.values)

# display sample with template
@app.route('/sample') 
def sample():
    sample = data_display.sample(10)
    return render_template("sample.html", table=sample.to_html(classes='data', index=False))

# display most recent 10 from api
# @app.route('/recent10')
# def recent10():
#     query = "SELECT body_length, channels, country, currency, delivery_method, description, email_domain, \
#                 fb_published, has_analytics, name, org_name, \
#                 sale_duration, user_age, venue_country, venue_name, created_at \
#                 FROM api_data WHERE created_at IS NOT NULL ORDER BY created_at DESC LIMIT 10;"
#     try:
#         rows = pd.read_sql(query, con=engine)
#     except:
#         return f"""something is broken"""
#     # return render_template('recent10.html', rows=[rows.to_html(
#     #     classes='table table-striped', index=False, table_id= 'recent10table')], titles=['na', rows.columns.values])
#     return render_template('recent10.html', data=rows.to_html( classes='table table-bordered', 
#                             index=False, table_id='dataTable', border=0))

# let user choose some parameters on what to query
@app.route('/query', methods=['GET', 'POST'])
def query():
    # form action is what to do when submitted
    return render_template("query.html")
    # return ''' enter the number of recent records from the API to view <form action="/query_results" method="POST">
    #             <input type="text" name="n_records" />
    #             <input type="submit" />
    #            </form>
    #          '''
@app.route('/test', methods=['GET', 'POST'])
def test():
    return render_template("test.html")

@app.route('/results', methods=['GET', 'POST'])
def results():
    try:
        n_sample = int(request.form['n_sample'])
        predict_type = request.form['predict_type']
        # return(str(predict_type))
        if predict_type == 'Correct Predictions':
            result = data_display[data_display['correct'] == 1]
        else:
            result = data_display[data_display['correct'] == 0]
        result = result.sample(n_sample)

        
        # get the image to display
        img_paths = [os.path.join(img_path, filename) for filename in result['filename']]
        img_paths = [filename for filename in result['filename']]
        # return str(img_paths)

        # create pretty output for predictions and don't show those cols
        predict_text = result['predict'].values[0]
        actual_text = result['actual'].values[0]
        result.drop(['predict', 'actual', 'correct', 'filename'], axis=1, inplace=True)

        # query = f"SELECT body_length, channels, country, currency, delivery_method, description, email_domain, \
        #         fb_published, has_analytics, name, org_name, \
        #         sale_duration, user_age, venue_country, venue_name, created_at \
        #         FROM api_data WHERE created_at IS NOT NULL ORDER BY created_at DESC LIMIT {n_records};"
        # rows = pd.read_sql(query, con=engine)
    except:
        return f"""You have entered an incorrect value or something isn't quite working right.
                    Sorry about that!  Hit the back button and try again."""

    return render_template('results.html', 
                            predict_text=predict_text, 
                            actual_text=actual_text, 
                            img_paths=img_paths,
                            data=result.to_html(index=False))

if __name__ == '__main__':
    # db_details = f'postgresql://postgres:{getpass()}@3.20.229.59:5432/campsite'
    # engine = setup_db(db_details)
    # run appç
    if len(sys.argv) > 1:
        MEDIA_FOLDER = '/home/ec2-user/github/media/images/'
        app.run(host='0.0.0.0', port=33507, debug=False)
    else:
        app.run(host='0.0.0.0', port=8080, debug=True)
    # for AWS
    # app.run(host='0.0.0.0', port=33507, debug=False)