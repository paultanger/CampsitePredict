from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from getpass import getpass

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


app = Flask(__name__, root_path='./') # template_folder = 'templates/')

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

@app.route('/results', methods=['GET', 'POST'])
def results():
    try:
        n_sample = int(request.form['n_sample'])
        predict_type = int(request.form['predict_type'])
        print(predict_type)
        if predict_type == 1:
            result = data_display[data_display['correct'] == 1]
        else:
            result = data_display[data_display['correct'] == 0]
        result = result.sample(n_sample)

        # get the image to display
        img_list = [filename for filename in result['filename']]

        # query = f"SELECT body_length, channels, country, currency, delivery_method, description, email_domain, \
        #         fb_published, has_analytics, name, org_name, \
        #         sale_duration, user_age, venue_country, venue_name, created_at \
        #         FROM api_data WHERE created_at IS NOT NULL ORDER BY created_at DESC LIMIT {n_records};"
        # rows = pd.read_sql(query, con=engine)
    except:
        return f"""something is broken"""

    return render_template('results.html', data=result.to_html(index=False))

if __name__ == '__main__':
    # setup api save to db
    # db_details = f'postgresql://postgres:{getpass()}@3.20.229.59:5432/campsite'
    # engine = setup_db(db_details)
    # run app
    app.run(host='0.0.0.0', port=8080, debug=True)
    # for AWS
    # app.run(host='0.0.0.0', port=33507, debug=False)