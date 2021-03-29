from flask import Flask, render_template, request, jsonify
from preprocess import randomforest

app = Flask('___name___',template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    features=[x for x in request.form.values()]
    sym1 = request.form['symptom1']
    sym2 = request.form['symptom2']
    sym3 = request.form['symptom3']
    sym4 = request.form['symptom4']
    p = randomforest(sym1,sym2,sym3,sym4)
    return render_template('index.html', prediction_text=' {}'.format(p))
