import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
import re
import pickle

#def preProcess_data(text): #cleaning the data
#    
#    text = text.lower()
#    new_text = re.sub('[^a-zA-z0-9\s]','',text)
#    new_text = re.sub('rt', '', new_text)
#    return new_text

app = FastAPI()

data = pd.read_csv('courseplandata_final.csv')
#tokenizer = Tokenizer(num_words=2000, split=' ')
#tokenizer.fit_on_texts(data['text'].values)



def my_pipeline(text): #pipeline
  #text_new = preProcess_data(text)
  #X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
  #X = pad_sequences(X, maxlen=28)
  numList = []
  textList = text.split(',')
  for text in textList:
    numList.append(int(text.strip()))  
  return numList


@app.get('/') #basic get view
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}



@app.get('/predict', response_class=HTMLResponse) #data input by forms
def take_inp():
    return '''<form method="post"> 
    <input type="text" maxlength="28" name="text" value="jointAngle,emg,strokeLevel,medCondtn in space in between format"/>  
    <input type="submit"/> 
    </form>'''



@app.post('/predict') #prediction on data
def predict(text:str = Form(...)): #input is from forms
    formattedVal = my_pipeline(text) #cleaning and preprocessing of the texts
    
    forest_from_saved = pickle.load(open('courseDurationModel.sav', 'rb'))
    
    #loaded_model = tf.keras.models.load_model('sentiment.h5') #loading the saved model
    #predictions = loaded_model.predict(clean_text) #making prediction
    df_z = pd.DataFrame(formattedVal)
    df_z = df_z.transpose()
    df_z.columns = ['jointAngle','emg','strokeLevel','medCondtn']
    prediction = forest_from_saved.predict(df_z)
#    print(prediction)
    #sentiment = int(np.argmax(predictions)) #index of maximum prediction
    #probability = max(predictions.tolist()[0]) #probability of maximum prediction
    #if sentiment==0: #assigning appropriate name to prediction
    #    t_sentiment = 'negative'
    #elif sentiment==1:
    #    t_sentiment = 'neutral'
    #elif sentiment==2:
    #    t_sentiment='postive'
    result = dict(enumerate(prediction.flatten(), 1))
    return result