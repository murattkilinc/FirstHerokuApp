from flask import Flask,render_template,request,redirect,url_for

import numpy as np
import pickle

filename='finalized_model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route("/")
def index():

    return render_template("index.html")                

@app.route("/predict",methods = ["POST"])
def predictApp():

    ph = request.form.get("ph") 
    hardness = request.form.get("hardness") 
    solids = request.form.get("solids") 
    chloramines = request.form.get("chloramines") 
    sulfate = request.form.get("sulfate") 
    conductivity = request.form.get("conductivity") 
    organic_carbon = request.form.get("organic_carbon") 
    trihalomethanes = request.form.get("trihalomethanes")
    turbidity = request.form.get("turbidity")

    notPotable = np.array([0])
    potable = np.array([1])

    real_values = np.array([ph,hardness,solids,chloramines,sulfate,conductivity,organic_carbon,trihalomethanes,turbidity]).reshape(1, -1)

    predict_LR = model.predict(real_values)

    return render_template("index.html", potable = potable, notPotable = notPotable, predict_LR = predict_LR )                
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug = True)