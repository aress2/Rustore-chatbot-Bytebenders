import gradio as gr
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urljoin
import torch

# Проверка доступности CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загрузка моделей и токенизаторов
qa_model_name = "DeepPavlov/rubert-base-cased-conversational"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name).to(device)

code_model_name = "microsoft/codebert-base"  # Пример модели для работы с кодом
code_tokenizer = AutoTokenizer.from_pretrained(code_model_name)
code_model = AutoModelForCausalLM.from_pretrained(code_model_name).to(device)

# Список URL для поиска
urls = [
        "https://www.rustore.ru/help/",
    "https://www.rustore.ru/help/#__docusaurus_skipToContent_fallback",
    "https://www.rustore.ru/help/users/about-rustore",
    "https://www.rustore.ru/help/users/about-rustore#__docusaurus_skipToContent_fallback",
    "https://www.rustore.ru/help/developers/",
    "https://www.rustore.ru/help/developers/#__docusaurus_skipToContent_fallback",
    "https://www.rustore.ru/help/sdk/",
    "https://www.rustore.ru/help/work-with-rustore-api/",
    "https://www.rustore.ru/help/guides/",
    "https://www.rustore.ru/help/en/developers/",
    "https://www.rustore.ru/help/developers/developer-account",
    "https://www.rustore.ru/help/developers/publishing-and-verifying-apps/",
    "https://www.rustore.ru/help/developers/monetization/",
    "https://www.rustore.ru/help/developers/advertising-and-promotion/",
    "https://www.rustore.ru/help/developers/tools/",
    "https://www.rustore.ru/help/developers/developer-statistics/",
    "https://www.rustore.ru/help/developers/vk-id-sdk",
    "https://www.rustore.ru/help/developers/check-apk-signature",
    "https://www.rustore.ru/help/developers/faq",
    "https://www.rustore.ru/help/users/authorization/why-vk-id",
    "https://www.rustore.ru/help/legal/terms-of-use/",
    "https://www.rustore.ru/help/legal/privacy-policy-users/",
    "https://www.rustore.ru/help/en/users/about-rustore",
    "https://www.rustore.ru/help/en/users/about-rustore#__docusaurus_skipToContent_fallback",
    "https://www.rustore.ru/help/en/users/",
    "https://www.rustore.ru/help/en/developers/developer-account",
    "https://www.rustore.ru/help/en/sdk/",
    "https://www.rustore.ru/help/en/work-with-rustore-api/api-subscription-payment",
    "https://www.rustore.ru/help/en/guides/",
    "https://www.rustore.ru/help/en/users/start/",
    "https://www.rustore.ru/help/en/users/authorization",
    "https://www.rustore.ru/help/en/users/app-management",
    "https://www.rustore.ru/help/en/users/purchases-and-returns",
    "https://www.rustore.ru/help/en/users/policies",
    "https://www.rustore.ru/help/en/",
    "https://www.rustore.ru/help/en/users/about-rustore#about-rustore",
    "https://www.rustore.ru/help/en/users/about-rustore#our-goal",
    "https://www.rustore.ru/help/en/users/about-rustore#how-to-download-and-install-rustore",
    "https://www.rustore.ru/help/en/users/policies/work-in-background",
    "https://www.rustore.ru/help/users/start",
    "https://www.rustore.ru/help/users/start#__docusaurus_skipToContent_fallback",
    "https://www.rustore.ru/help/en/users/start",
    "https://www.rustore.ru/help/users/start/app-install",
    "https://www.rustore.ru/help/users/start/install-on-tv",
    "https://www.rustore.ru/help/users/start/notification",
    "https://www.rustore.ru/help/users/start/interface",
    "https://www.rustore.ru/help/users/start/app-page",
    "https://www.rustore.ru/help/users/start/update-rustore-app",
    "https://www.rustore.ru/help/users/start/change-theme",
    "https://www.rustore.ru/help/users/start/parental-control",
    "https://www.rustore.ru/help/users/authorization",
    "https://www.rustore.ru/help/users/app-management",
    "https://www.rustore.ru/help/users/purchases-and-returns",
    "https://www.rustore.ru/help/users/policies",
    "https://www.rustore.ru/help/users/",
    "https://www.rustore.ru/help/users/about-rustore#о-rustore",
    "https://www.rustore.ru/help/users/authorization/login-vk-id",
    "https://www.rustore.ru/help/users/authorization/login-gosuslugi-id",
    "https://www.rustore.ru/help/users/authorization/login-sber-id",
    "https://www.rustore.ru/help/users/authorization/login-tinkoff-id",
    "https://www.rustore.ru/help/users/authorization/how-to-get-out",
    "https://www.rustore.ru/help/users/authorization/without-authorization",
    "https://www.rustore.ru/help/users/app-management/app-from-rustore",
    "https://www.rustore.ru/help/users/app-management/app-search",
    "https://www.rustore.ru/help/users/app-management/app-review",
    "https://www.rustore.ru/help/users/app-management/update-app-from-rustore",
    "https://www.rustore.ru/help/users/app-management/auto-update",
    "https://www.rustore.ru/help/users/app-management/vk-mini-apps",
    "https://www.rustore.ru/help/users/purchases-and-returns/paid-apps",
    "https://www.rustore.ru/help/users/purchases-and-returns/payment-method",
    "https://www.rustore.ru/help/users/purchases-and-returns/subscribe",
    "https://www.rustore.ru/help/users/purchases-and-returns/link-bank-card",
    "https://www.rustore.ru/help/users/purchases-and-returns/payment-history",
    "https://www.rustore.ru/help/users/purchases-and-returns/foreign-apps",
    "https://www.rustore.ru/help/users/purchases-and-returns/sber-id-connect",
    "https://www.rustore.ru/help/users/about-rustore/",
    "https://www.rustore.ru/help/users/about-rustore/#__docusaurus_skipToContent_fallback",
    "https://www.rustore.ru/help/users/about-rustore/#о-rustore",
]
# Функция для извлечения текста с сайта, включая вложенные ссылки
def extract_text_from_url(url, depth=0, max_depth=2, visited=None):
    if visited is None:
        visited = set()
    if depth > max_depth or url in visited:
        return []

    visited.add(url)
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join([p.text for p in soup.find_all('p')])
        result = [(url, text)]

        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            if full_url.startswith('https://www.rustore.ru/help/'):
                result.extend(extract_text_from_url(full_url, depth + 1, max_depth, visited))

        return result
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return []

# Загрузка контекста со всех указанных URL, включая вложенные ссылки
contexts = []
for url in urls:
    contexts.extend(extract_text_from_url(url))

# Функция для поиска наиболее релевантного фрагмента
def find_most_relevant(query, contexts):
    vectorizer = TfidfVectorizer()
    texts = [context[1] for context in contexts]
    contexts_vector = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, contexts_vector)[0]
    most_relevant_index = similarities.argmax()
    most_relevant_context = contexts[most_relevant_index][1]
    most_relevant_url = contexts[most_relevant_index][0]
    similarity_score = similarities[most_relevant_index]
    return most_relevant_context, most_relevant_url, similarity_score

# Функция для генерации ответа при поиске по сайтам
def generate_answer(query):
    relevant_context, relevant_url, similarity_score = find_most_relevant(query, contexts)
    inputs = qa_tokenizer.encode_plus(query, relevant_context, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Перемещаем входные данные на GPU
    with torch.no_grad():
        outputs = qa_model(**inputs)
    answer_start = outputs.start_logits.argmax().item()
    answer_end = outputs.end_logits.argmax().item() + 1
    answer = qa_tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
    return answer, relevant_url, similarity_score

# Функция для исправления кода
def fix_code(code):
    inputs = code_tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Перемещаем входные данные на GPU
    with torch.no_grad():
        outputs = code_model.generate(**inputs, max_length=512, num_return_sequences=1, temperature=0.7)
    fixed_code = code_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return fixed_code

# Новая функция для генерации кода на Kotlin
def generate_kotlin_code(query):
    url = "https://www.codeconvert.ai/kotlin-code-generator"
    try:
        response = requests.get(url, params={"query": query})
        soup = BeautifulSoup(response.content, 'html.parser')
        pre_tag = soup.find('pre')
        if pre_tag:
            code = pre_tag.text
        else:
            code = "Не удалось найти сгенерированный код."
    except Exception as e:
        code = f"Ошибка при получении кода: {str(e)}"
    return code

# Интерфейс Gradio
def gradio_interface(query, mode):
    if mode == "Генерация кода на Kotlin":
        code = generate_kotlin_code(query)
        return code, "Код генерирован"
    elif mode == "Поиск по сайтам":
        answer, url, similarity = generate_answer(query)
        return answer, f"Ссылка: {url}\nБлизость: {similarity:.2f}"
    elif mode == "Исправление кода":
        fixed_code = fix_code(query)
        return fixed_code, "Код исправлен"

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(lines=5, placeholder="Введите ваш запрос или код здесь..."),
        gr.Radio(["Генерация кода на Kotlin", "Поиск по сайтам", "Исправление кода"], label="Режим работы")
    ],
    outputs=["text", "text"],
    title="Помощник по разработке мобильных приложений для RuStore",
    description="Задайте вопрос о разработке мобильных приложений для RuStore, введите код для исправления или запросите генерацию кода на Kotlin"
)
iface.launch()
