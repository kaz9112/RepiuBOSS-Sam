import streamlit as st
import pandas as pd
import gensim
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, TextVectorization, GlobalAveragePooling1D, Input, LSTM, GRU, Dropout, Dense

from scraper import scrape
from func import TextProcess, Label, TextProcess2

lda_model_inf =  gensim.models.LdaModel.load('lda.model')

model = tf.keras.models.load_model('model_imp')

# with open('model_imp.pkl', 'rb') as file_1:
#   model = joblib.load(file_1)

with open('dictionary.pkl', 'rb') as file_4:
  dictionary_inf = joblib.load(file_4)

def run():
    # Membuat Title
    st.title('Negative Review clustering insights')
    

    # membuat form
    with st.form(key='form_parameters'):
        text_input = st.text_input('Reviews', 'Barangnya tidak sesuai gambar')
        submitted_text = st.form_submit_button('Show topic')

    data_inf = {
        'text': text_input
    }


    data_inf = pd.DataFrame([data_inf])
    
    if submitted_text:
        data_inf['text_processed'] = data_inf['text'].apply(lambda x: TextProcess(x))
        Message_body_inf = data_inf['text_processed']

        data_inf['text_processed_2'] = data_inf['text'].apply(lambda x: TextProcess2(x))
        Message_body_inf_2 = data_inf['text_processed_2']

        y_pred_inf = model.predict(Message_body_inf_2)
        y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)

        if y_pred_inf == 0:
            st.write('Negative sentiment')
            for i in range(len(Message_body_inf)):
                bow_vector = dictionary_inf.doc2bow(Message_body_inf[i])
                for index, score in sorted(lda_model_inf[bow_vector], key=lambda tup: -1*tup[1]):
                    # st.write(f"Score: {round(score*100)}%")
                    st.write(f"Complaint: {Label(index)}")
                    # st.write("Complaint: {}".format(Label(index)))
                    break
                st.write('------------------------------------')
        else:
            st.write('Positive sentiment')


    

if __name__ == '__main__':
    run()