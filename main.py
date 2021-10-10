#FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints

from fastapi import FastAPI
import pandas as pd
from pycaret.classification import load_model, predict_model


app = FastAPI()
db = pd.read_csv('data.csv') 
model = load_model('final_model') 


#app objekt and use a decorator (you can use a HTTP request (GET, POST, ...)  
#decorators: add functionality to an existing code.
@app.get('/historical')
def get_historical_data(): 
    js = db.to_json(orient = 'records')
    return js 


@app.get('/tomorrow')
def get_future_data(): 
    input_data =  pd.read_csv('new_data.csv')
    input_data.drop('cnt', axis=1, inplace=True)
    pred_df = predict_model(model, input_data)
    
    js = pred_df.to_json(orient = 'records')
    return js 

#run this localy in the terminal: uvicorn main:app --reload, 
#main is the filename, and the name of the app object
#http://127.0.0.1:8000/docs