import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,_tree
from fastapi import FastAPI, Request

MODEL_FNAME = "data/model.pkl"

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(request: Request):
    req = await request.json()
    list_s = req["symptoms"]
    data = {"success": False}

    symptoms = []
    dados = pd.read_csv("data/dados-traduzidos.csv")
    cols = dados.columns
    for idx,value in enumerate(dados[cols[3]]):
        if value in list_s:
            symptoms.append(dados[cols[2]][idx])

    result = sec_predict(symptoms)
    print(result)
    for idx,value in enumerate(dados[cols[0]]):
        if value ==  result[0]:
            result[0] = dados[cols[1]][idx]
            break

    data["response"] = result[0]
    data["success"] = True

    return data

def sec_predict(symptoms_exp):
    print(symptoms_exp)
    df = pd.read_csv('data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])
