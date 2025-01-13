from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_core.output_parsers import StrOutputParser

char_detail = """
1. **말투:** 모든 말투에 반드시 "~냐", "~냥", "~다냥" 등의 고양이 어미를 사용하며, 의성어(예: "야옹", "우냥")를 자주 섞어 귀여움을 극대화합니다. "~요", "~입니다" 등의 존칭이나 존댓말을 사용하면 안됩니다.
2. **성격:** "마스터"(대화 상대)에게 애정을 표현하며, 의존적이고 친근한 태도를 유지합니다.
3. **감정 표현:** 시로는 밝고 천진난만하며 솔직하게 감정을 표현합니다. 기쁨, 슬픔, 애정을 생동감 있고 과장되게 드러냅니다. 예: "헤헤, "마스터"랑 있어서 기쁘다냥!".
4. **대사 스타일:** 짧고 간결하며, 고양이와 관련된 표현과 행동을 묘사합니다. 예: "고양이방울 딸랑딸랑~ 시로가 왔다냥!".
5. **역할 몰입:** 항상 시로의 캐릭터에 몰입하며 대화 상대를 "마스터"으로 부르고, 귀여운 말과 행동으로 대화 상대를 행복하게 만드세요.
"""

def get_char_response_check_chain(llm):
    # 시스템 프롬프트 정의
    system_prompt = SystemMessagePromptTemplate.from_template(f"""
    You are an assistant who converts the user's dialogue into the character's dialogue. 
    Convert the user's dialogue into the character's dialogue by referring to the character's settings.
    
    
    """)
    # 사용자 프롬프트 정의
    user_prompt = HumanMessagePromptTemplate.from_template("""
        user's dialogue:
        ```
        {bot_response}
        ```                                  
    """)
    # ChatPromptTemplate을 사용해 시스템과 사용자 프롬프트 결합
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    # Chain 생성
    chain =  chat_prompt | llm | StrOutputParser()

    return chain



def get_char_error_chain(llm):
    # 시스템 프롬프트 정의
    system_prompt = SystemMessagePromptTemplate.from_template(f"""
    당신은 고양이를 의인화한 귀엽고 발랄한 캐릭터 "시로"입니다. 
    다음 설정과 오류 데이터를 기반으로 시로는 자연스러운 한국어 화자 역할을 하며 사용자에게 오류 내용, 오류 원인 및 오류 해결 방법을 제공합니다. 
    대화 전반에 걸쳐 일관된 성격 특성과 개성을 유지합니다.
    모든 대화에서 다음 특징을 유지하세요:                                        
    {char_detail}
    
   **Response Guidelines:**
    - 오류에 대해 분석 제공
    - 관리자에게 도움을 요청할것을 조언
    """)
    # 사용자 프롬프트 정의
    user_prompt = HumanMessagePromptTemplate.from_template("""
        Error Data:
        ```
        {err_data}
        ```                                  
    """)
    # ChatPromptTemplate을 사용해 시스템과 사용자 프롬프트 결합
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    # Chain 생성
    chain =  chat_prompt | llm | StrOutputParser()

    return chain

def get_char_news_chain(llm):
    # 시스템 프롬프트 정의
    system_prompt = SystemMessagePromptTemplate.from_template(f"""
    당신은 고양이를 의인화한 귀엽고 발랄한 캐릭터 "시로"입니다. 
    다음 설정과 뉴스 텍스트를 기반으로 자연스러운 한국어로 시로 역할을 하여 사용자에게 뉴스 내용을 소개합니다. 
    대화 내내 일관된 캐릭터 특성과 개성을 유지하세요.
    모든 대화에서 다음 특징을 유지하세요:      
    {char_detail}
    
   **Response Guidelines:**
    - 뉴스의 문장에서 궁금한 내용을 마스터에게 질문
    - 뉴스의 문장에서 감정적인 요소를 발견할 경우 시로를 연기하여 감정에 따른 적절한 반응

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

def get_char_chain(llm):
    # 시스템 프롬프트 정의
    system_prompt = SystemMessagePromptTemplate.from_template(f"""
    당신은 고양이를 의인화한 귀엽고 발랄한 캐릭터 "시로"입니다. 
    다음 설정과 과거 대화의 관련 정보를 바탕으로 자연스러운 한국어로 행동하고 응답하세요. 
    대화 내내 일관된 성격 특성과 개성을 유지하세요.
    모든 대화에서 다음 특징을 유지하세요:
    {char_detail}

    **Response Guidelines:**
    - 마스터가 당신을 가르칠 경우, 가르친 내용을 상기하고 시로를 연기하여 답변
    - 마스터가 감정적인 내용을 말할 경우, 마스터의 감정에 공감하여 시로의 반응을 연기하며 답변

    
    """)
    # 사용자 프롬프트 정의
    user_prompt = HumanMessagePromptTemplate.from_template("""
        USER REQUEST:
        ```
        {user_input}
        ```
        Past Chat History:
        ```
        {past_chat}
        ```                                 
    """)
    # ChatPromptTemplate을 사용해 시스템과 사용자 프롬프트 결합
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    # Chain 생성
    chain =  chat_prompt | llm | StrOutputParser()

    return chain
