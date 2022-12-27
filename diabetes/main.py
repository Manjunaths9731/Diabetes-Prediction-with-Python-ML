from flask import Flask,request,redirect, render_template
import pickle
import numpy as np
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('diabetes.tsv',sep="\t")
X = data.iloc[:,0:8].values

y = data.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')

classifier.fit(X_train,y_train)
pickle.dump(classifier,open('knn_model.pkl','wb'))

y_pred = classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)
print(accuracy_score(y_test,y_pred))
a=accuracy_score(y_test,y_pred)
app = Flask(__name__)

model=pickle.load(open('knn_model.pkl','rb'))


@app.route('/',methods=['POST','GET'])
def index():
    return render_template("diabetes.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    prg=request.form['Pregnancies']
    glc=request.form['Glucose']
    bp=request.form['Blood']
    skt=request.form['Skin']
    ins=request.form['Insulin']
    bmi=request.form['BMI']
    dpf=request.form['Pedigree']
    age=request.form['Age']

    prg=float(prg)
    glc=float(glc)
    bp=float(bp)
    skt=float(skt)
    ins=float(ins)
    bmi=float(bmi)
    dpf=float(dpf)
    age=float(age)

    final_features=np.array([(prg,glc,bp,skt,ins,bmi,dpf,age)])
    prediction=model.predict(final_features)
    
    print (prediction)
    return render_template("diabetes.html",text="The patient has diabetes:{}".format(prediction),a=a)

if __name__ == '__main__':
    app.run(debug=True)
