
from transformers import pipeline
from streamlit_chat import message
import streamlit as st
import warnings
warnings.simplefilter("ignore")

sentiment_label = {'LABEL_0':"부정", 'LABEL_1':"긍정"}

if 'model' not in st.session_state:
    model = pipeline("text-classification", 
                    model="matthewburke/korean_sentiment")
    st.session_state['model'] = model

st.title("🤖 감정분석 LLM")
st.markdown('---')

with st.form('chat', clear_on_submit=True):
    user_prompt = st.text_input('프롬프트 입력:')
    submit_button = st.form_submit_button(label='확인')
    if user_prompt and submit_button:
        with st.spinner('GPT가 답변을 준비하는 중...'):
            result =st.session_state['model'](user_prompt)[0]['label']
            st.write('감정:', sentiment_label[result] )
            st.write('확률:', st.session_state['model'](user_prompt)[0]['score'])
