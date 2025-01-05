import requests
from lxml import etree
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def fetch_content(url: str, title_xpath: str, content_xpath: str):
    """
    주어진 URL에서 제목과 본문 데이터를 추출하여 객체로 반환하는 함수.

    :param url: 대상 웹페이지의 URL
    :param title_xpath: 제목이 위치한 XPath
    :param content_xpath: 본문이 위치한 XPath
    :return: {"title": 제목 문자열, "content": 본문 객체}
    """
    # 1) URL에서 HTML 내용 가져오기
    response = requests.get(url)
    html_content = response.content

    # 2) HTML 내용을 lxml.etree로 파싱
    root = etree.HTML(html_content)

    # 3) 제목 추출
    title_nodes = root.xpath(title_xpath)
    if title_nodes:
        title = "".join(node.strip() for node in title_nodes[0].itertext() if node.strip())
    else:
        title = "제목 없음"

    # 4) 본문 추출
    content_nodes = root.xpath(content_xpath)
    if not content_nodes:
        return {"title": title, "content": {"text": "", "images": []}}

    # 첫 번째 매칭된 본문 노드 선택
    content_node = content_nodes[0]

    # 본문 HTML 변환
    content_html = etree.tostring(content_node, encoding='utf-8').decode('utf-8')

    # BeautifulSoup으로 파싱
    soup = BeautifulSoup(content_html, 'html.parser')

    # 불필요한 태그들 제거
    for element in soup.find_all(['script', 'style', 'iframe', 'a', 'img']):
        element.decompose()

    # 본문 텍스트 추출 - get_text() 메소드 사용하여 순수 텍스트만 추출
    text = soup.get_text(separator=' ', strip=True)

    # 이미지 URL 추출
    images = [img.get('src') for img in soup.find_all('img') if img.get('src')]

    # 본문 객체 구성
    content = {
        "text": text,
        "images": images
    }

    return {"title": title, "content": content,"site_url": url}


def get_a_tags_from_xpath(url: str, xpath: str):
    """
    주어진 URL의 HTML 문서를 파싱하여, 지정된 XPath에 해당하는 노드의
    1단계 자식들을 순회한 뒤, 각 자식 노드의 모든 자손들이 가진 <a> 태그의
    (href, 텍스트) 정보를 하나의 리스트로 취합하여 반환합니다.

    :param url: 대상 웹페이지의 URL
    :param xpath: 추출할 HTML 노드의 XPath
    :return: 각 <a> 태그에 대한 정보를 담고 있는 사전(dict)들의 리스트
    """
    # 1) URL에서 HTML 내용 가져오기
    response = requests.get(url)
    html_content = response.content  # .content는 bytes, .text는 str

    # 2) lxml.etree를 사용하여 문서를 파싱하고 XPath로 특정 노드 찾기
    root = etree.HTML(html_content)
    target_nodes = root.xpath(xpath)

    # 만약 해당 XPath가 매칭되는 노드가 없다면 빈 리스트 반환
    if not target_nodes:
        return []

    # 3) 첫 번째 매칭 노드만 사용 (여러 노드가 매칭될 경우 필요에 맞게 조정)
    target_node = target_nodes[0]

    # 4) 찾은 노드를 문자열(HTML)로 변환 후 BeautifulSoup으로 재파싱
    target_node_html = etree.tostring(target_node, encoding='utf-8').decode('utf-8')
    soup = BeautifulSoup(target_node_html, 'html.parser')

    # 5) 해당 노드(가장 바깥 노드)의 직계 자식 찾기
    container = soup.find()  # 재파싱한 뒤 최상단 노드를 하나 찾음
    if not container:
        return []

    direct_children = container.find_all(recursive=False)

    # 6) 각 자식 노드의 모든 자손에서 <a> 태그의 링크와 텍스트를 수집
    result_list = []
    for child in direct_children:
        # 모든 자손 노드에 대해 a 태그 찾기
        a_tags = child.find_all('a')
        for a in a_tags:
            # url = 'https://www.aitimes.com/news/articleList.html?page=1&total=4296&box_idxno=&sc_sub_section_code=S2N48'
            link = a.get('href')
            if link:
                # urljoin()은 base_url과 상대 경로를 지능적으로 결합
                # 이미 완전한 URL이면 그대로 유지, 상대 경로면 base_url과 결합
                absolute_link = urljoin(url, link)
                text = a.get_text(strip=True)
                result_list.append({
                    'link': absolute_link,
                    'text': text
                })

    return result_list

def fetch_page_results(site_item,page_number: int):
    """
    페이지 번호를 입력받아 결과 데이터를 반환하는 함수.

    :param page_number: 대상 페이지 번호
    :return: 각 항목의 상세 데이터를 포함하는 리스트
    """

    # 환경 변수에서 URL과 XPath 불러오기 2
    base_url = site_item['base_url']
    item_list_xpath = site_item['item_list_xpath']
    title_xpath = site_item['title_xpath']
    content_xpath = site_item['content_xpath']

    # 페이지 URL 구성
    page_url = base_url.format(page=page_number)

    # 항목 리스트 가져오기
    extracted_data = get_a_tags_from_xpath(page_url, item_list_xpath)
    result_array = []

    # 각 항목의 상세 데이터 가져오기
    for item in extracted_data:
        detail_page = fetch_content(item['link'], title_xpath, content_xpath)
        result_array.append(detail_page)

    return result_array


def get_ai_times_data(page_num = 1):
    ai_times = {
        'base_url': "https://www.aitimes.com/news/articleList.html?page={page}&total=4296&box_idxno=&sc_sub_section_code=S2N48",
        'item_list_xpath': '//*[@id="section-list"]',
        'title_xpath': '//*[@id="articleViewCon"]/article/header/h3',
        'content_xpath': '//*[@id="article-view-content-div"]',
    }
    data = fetch_page_results(ai_times,page_num)
    return data


if __name__ == "__main__":
    inven = {
        'base_url': "https://www.inven.co.kr/webzine/news/?site=duck&iskin=duck&vtype=pc&page={page}",
        'item_list_xpath': '//*[@id="webzineNewsListF1"]/div[3]/div/table/tbody',
        'title_xpath': '//*[@id="webzineNewsView"]/div[2]/div[2]',
        'content_xpath': '//*[@id="imageCollectDiv"]',
    }    
    ai_times = {
        'base_url': "https://www.aitimes.com/news/articleList.html?page={page}&total=4296&box_idxno=&sc_sub_section_code=S2N48",
        'item_list_xpath': '//*[@id="section-list"]',
        'title_xpath': '//*[@id="articleViewCon"]/article/header/h3',
        'content_xpath': '//*[@id="article-view-content-div"]',
    }
    test = fetch_page_results(ai_times,1)
    print(test[0])