from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_core.output_parsers import StrOutputParser

def get_summury_chain(llm):
    # 시스템 프롬프트 정의
    system_prompt = SystemMessagePromptTemplate.from_template("""
        You are an advanced language model specialized in summarizing Korean news articles. Read the following Korean news article and summarize its content according to the conditions below:

        1. **Key Content Summary**: Concisely summarize the most important facts and content of the article. Exclude unnecessary details.
        2. **Sentence Limit**: Write the summary in 3 to 5 sentences.
        3. **Objective Expression**: Avoid personal opinions or emotions and deliver the content in a neutral tone.
        4. **Clarity and Conciseness**: Write clear and concise sentences that are easy for readers to understand.
        
    """)
    # 사용자 프롬프트 정의
    user_prompt = HumanMessagePromptTemplate.from_template("""
        News Article Input:
        ```
        {news_article}
        ```                              
    """)
    # ChatPromptTemplate을 사용해 시스템과 사용자 프롬프트 결합
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    # Chain 생성
    chain =  chat_prompt | llm | StrOutputParser()

    return chain
