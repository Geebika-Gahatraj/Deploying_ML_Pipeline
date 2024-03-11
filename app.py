from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# binary_classifier = pickle.load(open('models/binaryclassifier.pkl','rb'))
# multiclass_classifier = pickle.load(open('models/multiclass.pkl','rb'))
ensemble_model = pickle.load(open('models/ensemble.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/binary_predict', methods=['POST'])
def binary_predictor():
    founded_at = int(request.form.get('founded_at'))
    first_funding_at = int(request.form.get('first_funding_at'))
    last_funding_at = int(request.form.get('last_funding_at'))
    funding_rounds = float(request.form.get('funding_rounds'))
    funding_total_usd = float(request.form.get('funding_total_usd'))
    first_milestone_at = int(request.form.get('first_milestone_at'))
    last_milestone_at = int(request.form.get('last_milestone_at'))
    milestones = float(request.form.get('milestones'))
    relationships = float(request.form.get('relationships'))

    input_data = {
        'founded_at': founded_at,
    'first_funding_at' : first_funding_at,
    'last_funding_at': last_funding_at,
    'funding_rounds' : funding_rounds,
    'funding_total_usd': funding_total_usd,
    'first_milestone_at': first_milestone_at,
    'last_milestone_at' : last_milestone_at,
    'milestones' : milestones,
    'relationships': relationships
    }
    input_df =pd.DataFrame([input_data])
    
    result=ensemble_model.predict(input_df)[0]
    

    # print the predicted output
    print(f"Predicted output from binary classifier : {result}")

    class_mapping = {0 :'Operating', 1 : 'Acquired' , 2: 'Closed', 3 : 'IPO'}
    predicted_label = class_mapping[result]
    print("Predicted classes:",predicted_label)
    


    return render_template('results.html',predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)

