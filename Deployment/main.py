import streamlit as st
import pandas as pd
import gensim
import joblib

from func import TextProcess, Label

lda_model_inf =  gensim.models.LdaModel.load('lda.model')

with open('dictionary.pkl', 'rb') as file_4:
  dictionary_inf = joblib.load(file_4)

def run():
    # Membuat Title
    st.title('Negative Review clustering insights')

    # membuat form
    with st.form(key='form_parameters'):
        text_input = st.text_input('Reviews', 'Barangnya tidak sesuai gambar')
        submitted = st.form_submit_button('Predict')

    data_inf = {
        'text': text_input
    }

    data_inf = pd.DataFrame([data_inf])

    if submitted:
        data_inf['text_processed'] = data_inf['text'].apply(lambda x: TextProcess(x))
        
        Message_body_inf = data_inf['text_processed']
        
        if sentiment == 0:
            for i in range(len(Message_body_inf)):
                bow_vector = dictionary_inf.doc2bow(Message_body_inf[i])
                for index, score in sorted(lda_model_inf[bow_vector], key=lambda tup: -1*tup[1]):
                    st.write(f"Score: {round(score*100)}%")
                    st.write(f"Complaint: {Label(index)}")
                    # st.write("Complaint: {}".format(Label(index)))
                    break
                st.write('------------------------------------')
        else:
            st.write('Sentimen positif')

if __name__ == '__main__':
    run()