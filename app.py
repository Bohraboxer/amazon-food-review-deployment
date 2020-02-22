from flask import Flask,request,jsonify,render_template,url_for

import packages_to_import as p
from understanding_data import understanding_data
import cleaning_and_preparing_train_data
import training
from training import *


# we can add the training data and uncomment the below lines
# data=p.pd.read_csv("Reviews.csv")
# data = data.sample(frac =0.6)


# to undersatnd the data and for cleaning of training data we can uncomment the below code
#understanding_data(data)
#cleaning_and_preparing_train_data.df(data)
# summary_query=input("Enter the summary")
# text_query=input("Enter the review")



app=Flask(__name__)


@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    if request.method == 'POST':
        message=request.form['Text']
        test_data = p.pd.DataFrame(p.np.column_stack([message]),columns=['Text'])
        tt = train_test(test_data)
        tt.testing()
        my_prediction = tt.testing()
        my_prediction = 1 if my_prediction >= 0.5 else 0
    return render_template('result.html',prediction=my_prediction)


if __name__ =="__main__":
    app.run(use_reloader=True, use_debugger=True,host="0.0.0.0",threaded=False)