import streamlit as st
import pandas as pd
import gensim
import joblib
from scraper import scrape
import requests
import numpy as np
import json

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
        submitted_text = st.form_submit_button('Show topic')

    data_inf = {
        'text': text_input
    }
    
    # with st.form(key='form_parameters'):
    #     url_text = st.text_input('Url link review')
    #     url_final = url_text
    #     submitted_list = st.form_submit_button('Show topic numerical')


    data_inf = pd.DataFrame([data_inf])
    

    
    if submitted_text:
        data_inf['text_processed'] = data_inf['text'].apply(lambda x: TextProcess(x))
        Message_body_inf = data_inf['text_processed'][0].tolist()
        print(Message_body_inf)

        # URL = "https://tf-serving-backend-group3.herokuapp.com/v1/models/sentiment_model:predict"
        URL = "https://tf-serving-backend-group3.herokuapp.com/v1/models/sentiment_model"

        input_data_json = json.dumps({
            'signature_name': 'serving_default',
            'instances': Message_body_inf
        })

        # r = requests.post(URL, data=input_data_json)
        r = requests.get(URL)
        st.write(r.status_code)


    #     if r.status_code == 200:
    #         data_inf['text_processed'] = data_inf['text'].apply(lambda x: TextProcess(x))
    #         Message_body_inf = data_inf['text_processed']

    #         for i in range(len(Message_body_inf)):
    #             bow_vector = dictionary_inf.doc2bow(Message_body_inf[i])
    #             for index, score in sorted(lda_model_inf[bow_vector], key=lambda tup: -1*tup[1]):
    #                 st.write(f"Score: {round(score*100)}%")
    #                 st.write(f"Complaint: {Label(index)}")
    #                 # st.write("Complaint: {}".format(Label(index)))
    #                 break
    #             st.write('------------------------------------')
    #         res = r.json()
    #         st.write(res['predictions'][0][0])
    #         # if res['predictions'][0][0] <= 0.5:
    #         #     st.write('Apple')
    #         # else:
    #         #     st.write('Orange')
    #     else:
    #         st.write('Error, reload page or retry predictions')

    
    # label_list = []
    # if submitted_list:
    #     data_inf = scrape(url_final)
    #     data_inf['text_processed'] = data_inf['text'].apply(lambda x: TextProcess(x))
    #     Message_body_inf = data_inf['text_processed']

    #     for i in range(len(Message_body_inf)):
    #         bow_vector = dictionary_inf.doc2bow(Message_body_inf[i])
    #         for index, score in sorted(lda_model_inf[bow_vector], key=lambda tup: -1*tup[1]):
    #             label_list.append(Label(index))
    #             # st.write(f"Score: {round(score*100)}%")
    #             # st.write(f"Complaint: {Label(index)}")
    #             # st.write("Complaint: {}".format(Label(index)))
    #             break
    #     dict_1 = {
    #         'text': label_list
    #     }
    #     df_1 = pd.DataFrame(dict_1)
    #     counts = df_1['text'].value_counts()
    #     percent100 = df_1['text'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
    #     df_final = pd.DataFrame({'counts': counts, 'per100': percent100})

    #     st.write(df_final)

if __name__ == '__main__':
    run()