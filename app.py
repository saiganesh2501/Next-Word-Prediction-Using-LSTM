import streamlit as st
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model=load_model('Lstm_model.keras')

#Import tokenizer pickle file
with open('tokenizer.pkl','rb')as file:
    tokenizer=pickle.load(file)

with open("hamlet.txt", "r", encoding="utf-8") as f:
    hamlet_text = f.read()

st.sidebar.title("Hamlet Text File")

st.sidebar.text_area(
    label="Hamlet Text File",
    value=hamlet_text,
    height=600,
    label_visibility="collapsed"
)  


## function to predict the next word
def predict_next_word(model,tokenizer,text,max_length_sequence):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_length_sequence:
        token_list=token_list[-(max_length_sequence-1):]
    token_list=pad_sequences([token_list],maxlen=max_length_sequence-1,padding='pre')
    prediction=model.predict(token_list,verbose=0)
    prediction_word_index=np.argmax(prediction,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==prediction_word_index:
            return word
    return None

## Streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the Text from Hamlet.txt")

if st.button("Predict Next Word"):
    max_length_sequence=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,input_text,max_length_sequence)
    st.write(f"Input Text is: {input_text}")

    st.write(f"Predicted_Next_Word is: {next_word}")
