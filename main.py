import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model 

#step 1
#load IMDB dataset
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

#load the trained model
model = load_model( "simplernn_imdb.h5")
    

#step 2
#load Helper functions
#function to decode reviews i takes list of numbers and converts back into words.
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

#step 3
#Function to preprocess the input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review




#step 4
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive/negative):")

#user input
user_input = st.text_area("Movie Review")
if st.button("Classify"):
    preprocessed_input = preprocess_text(user_input)
    #Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction[0][0] >= 0.5 else 'negative'
  

#Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')

else:

   st.write("Please enter a review and click 'Classify' to see the sentiment prediction.")


