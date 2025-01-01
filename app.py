# streamlit run app.py
# LLM 테스트는 대부분 이쪽
import streamlit as st
import os, char_ratio
from langchain_community.chat_models import ChatOllama  # OpenAI LLM 사용 (필요에 따라 다른 LLM으로 변경 가능)

# Streamlit 애플리케이션 구현
st.title("베리타스·레이시오 채팅 애플리케이션")
st.write("""
    캐릭터 베리타스·레이시오와 대화를 나눠보세요. 이 캐릭터는 냉소적이고 권위적인 지식 추구자입니다.
""")


base_url = os.getenv('BASE_URL')
# exaone3.5:32b-instruct-q4_K_M gemma2:27b-instruct-q6_K phi3:14b-medium-128k-instruct-q8_0 solar-pro:22b-preview-instruct-q8_0, llama3 benedict/linkbricks-gemma2-27b-korean-advanced-q4:latest
# 왜 젬마가 제일 품질이 나을까?
# 시스템 프롬프트가 아예 안먹는 경우가 있음. solar-pro가 대표적
high_level_model = ChatOllama(base_url=base_url, model="gemma2:27b-instruct-q6_K", temperature=0.1, num_predict = 32768)
chain = char_ratio.get_char_chain(high_level_model)

# 사용자 입력
user_input = st.text_input("질문을 입력하세요:")

if user_input:
    with st.spinner("응답을 생성 중입니다..."):
        response = chain.invoke({"user_input": user_input})
    st.write("### 응답:")
    st.write(response)
