Локальная RAG-система на Python
!(https://img.shields.io/badge/ChromaDB-latest-purple?style=flat-square&amp;logo=chromadb)

🚀 Введение
Это приложение представляет собой простую, но функциональную реализацию Retrieval-Augmented Generation (RAG) системы на Python. Оно позволяет вам задавать вопросы на естественном языке, и система будет использовать ваши собственные статьи (хранящиеся в локальном JSON-файле) для поиска релевантной информации и генерации контекстно-ориентированных ответов с помощью большой языковой модели (LLM).

Ключевые особенности:

Локальная обработка данных: Все ваши статьи и векторная база данных хранятся и обрабатываются локально, обеспечивая конфиденциальность данных.
Гибкость LLM: Система по умолчанию использует локальную LLM через Ollama, но легко настраивается для работы с облачными LLM (например, OpenAI, Google Gemini, Anthropic Claude).
Модульная архитектура: Построена с использованием фреймворка LangChain, что делает её расширяемой и простой для понимания.
🧠 Архитектура RAG-системы
Приложение реализует стандартный пайплайн RAG, который состоит из следующих основных этапов:

Прием данных (Data Ingestion): Загрузка необработанных данных (статей) из локального JSON-файла.    
Предварительная обработка (Preprocessing): Разбиение больших статей на более мелкие, управляемые "фрагменты" (chunks) для эффективного извлечения и соответствия ограничениям LLM.    
Генерация эмбеддингов (Embedding Generation): Преобразование текстовых фрагментов в числовые векторные представления (эмбеддинги), которые улавливают их семантическое значение.    
Векторная база данных (Vector Database): Хранение этих эмбеддингов вместе с исходными текстовыми фрагментами для быстрого поиска по сходству.    
Извлечение (Retrieval): При получении запроса пользователя, поиск наиболее релевантных фрагментов в векторной базе данных на основе семантического сходства.    
Аугментация (Augmentation): Объединение извлеченных релевантных фрагментов с исходным запросом пользователя, чтобы предоставить LLM необходимый контекст.    
Генерация (Generation): Использование LLM для синтеза связного и информативного ответа на основе дополненного запроса.    
⚙️ Предварительные требования
Перед запуском приложения убедитесь, что у вас установлены и настроены следующие компоненты:

3.1. Python
Версия: Рекомендуется Python 3.9 или выше.    
Установка: Скачайте и установите Python с официального сайта Python.
3.2. Ollama (для локальной LLM)
Ollama — это инструмент командной строки, который позволяет запускать различные модели LLM с открытым исходным кодом (например, Llama 3.1, Gemma, Mistral) на вашей локальной машине.    

Загрузка и установка Ollama:
Перейдите на официальный сайт Ollama.
Загрузите и установите приложение Ollama для вашей операционной системы (macOS, Windows, Linux).
Загрузка модели LLM:
После установки Ollama, откройте терминал (или командную строку).
Выполните команду для загрузки модели llama3.1 (рекомендуется для начала) :bash ollama run llama3.1
Эта команда загрузит модель на ваш компьютер. Оставьте этот терминал открытым, пока работает ваше RAG-приложение.
  
Устранение ollama: command not found: Если вы столкнулись с этой ошибкой, убедитесь, что Ollama установлен корректно и добавлен в системную переменную PATH. Попробуйте перезапустить терминал или компьютер.
Windows: Проверьте, что C:\Program Files\Ollama добавлен в PATH.
macOS: which ollama должен показать /usr/local/bin/ollama.
Linux: Используйте curl -fsSL https://ollama.com/install.sh | sh для установки.
3.3. Python-библиотеки
Рекомендуется использовать виртуальное окружение для управления зависимостями.

Создайте виртуальное окружение (в корневой директории проекта):
Bash

python -m venv venv
Активируйте виртуальное окружение:
На Windows: .\venv\Scripts\activate
На macOS/Linux: source venv/bin/activate
Установите необходимые библиотеки:
Bash

pip install langchain langchain_community langchain-ollama langchain-chroma sentence-transformers jq
langchain: Основной фреймворк для создания приложений на основе LLM.    
langchain_community: Содержит интеграции для различных компонентов.    
langchain-ollama: Интеграция LangChain с Ollama для локальных LLM.    
langchain-chroma: Интеграция LangChain с ChromaDB, локальной векторной базой данных.    
sentence-transformers: Используется для генерации высококачественных эмбеддингов текста.    
jq: Python-пакет, используемый JSONLoader для эффективного парсинга JSON.    
📂 Структура проекта
Ваш проект должен содержать два основных файла:

your_articles.json: Содержит ваши статьи в формате JSON.
rag_app.py: Основной Python-скрипт, реализующий RAG-систему.
chroma_db/ (будет создан автоматически): Директория для хранения вашей векторной базы данных ChromaDB.    
📝 Настройка данных (your_articles.json)
Этот файл служит вашей базой знаний. Он должен быть в формате JSON, где каждая статья является объектом в корневом массиве.

Пример структуры your_articles.json:

JSON


article_id: Уникальный идентификатор статьи.
title: Заголовок статьи.
author: Автор статьи.
publication_date: Дата публикации.
category: Категория статьи.
full_content: Полный текст статьи. Это основное содержимое, которое будет использоваться для извлечения информации.
7. Исходный код (rag_app.py)
Файл rag_app.py содержит всю логику RAG-системы, разделенную на логические шаги.

Python

import json
from pathlib import Path
import os # Добавляем импорт os для работы с переменными среды

# Импорт компонентов LangChain
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Импорты для различных LLM (раскомментируйте нужные)
from langchain_ollama.llms import OllamaLLM # Для локальной Ollama
# from langchain_openai import ChatOpenAI # Для OpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI # Для Google Gemini
# from langchain_anthropic import ChatAnthropic # Для Anthropic Claude

print("Начало настройки RAG-системы...")

# --- 1. Прием данных: Загрузка статей из локальных JSON-файлов ---
# Путь к вашему JSON-файлу со статьями
file_path = './your_articles.json'

# Определяем функцию метаданных для извлечения конкретных полей из каждой записи статьи [23]
def extract_article_metadata(record: dict, metadata: dict) -> dict:
    metadata["article_id"] = record.get("article_id")
    metadata["title"] = record.get("title")
    metadata["author"] = record.get("author")
    metadata["publication_date"] = record.get("publication_date")
    metadata["category"] = record.get("category")
    return metadata

try:
    # Инициализируем JSONLoader для обработки каждого объекта статьи [23]
    # jq_schema '.[ ]' выбирает каждый объект в корневом массиве JSON
    # content_key указывает 'full_content' как основной текст для Document [23]
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.[ ]',
        content_key='full_content',
        metadata_func=extract_article_metadata
    )

    # Загружаем документы
    articles_documents = loader.load()
    print(f"Шаг 1: Загружено {len(articles_documents)} объектов LangChain Document из JSON-файла.")
    if articles_documents:
        print(f"   Пример содержимого первого документа (первые 100 символов): {articles_documents.page_content[:100]}...")
        print(f"   Пример метаданных первого документа: {articles_documents.metadata}")
    else:
        print("   В JSON-файле не найдено документов для загрузки. Убедитесь, что 'your_articles.json' не пуст.")
        exit() # Выходим, если нет документов для обработки
except FileNotFoundError:
    print(f"Ошибка: Файл {file_path} не найден. Убедитесь, что 'your_articles.json' находится в той же директории.")
    exit()
except json.JSONDecodeError:
    print(f"Ошибка: Не удалось декодировать JSON из {file_path}. Проверьте формат файла.")
    exit()
except Exception as e:
    print(f"Произошла ошибка при загрузке JSON: {e}")
    exit()

# --- 2. Предварительная обработка документов: Эффективное разбиение текста на фрагменты ---
# Инициализируем RecursiveCharacterTextSplitter [11, 3, 25, 4, 26]
# chunk_size определяет максимальный размер каждого фрагмента [3, 4, 26]
# chunk_overlap определяет количество символов, которые будут перекрываться между соседними фрагментами [4, 27, 26]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Оптимальный размер фрагмента зависит от LLM и данных
    chunk_overlap=200, # Рекомендуется 10-20% перекрытия [27]
    separators=["\n\n", "\n", " ", ""] # По умолчанию, но можно настроить [3, 4]
)

# Разбиваем загруженные документы на фрагменты
chunked_documents = text_splitter.split_documents(articles_documents)
print(f"Шаг 2: Создано {len(chunked_documents)} фрагментов из исходных документов.")
if chunked_documents:
    print(f"   Пример содержимого первого фрагмента (первые 200 символов): {chunked_documents.page_content[:200]}...")
    print(f"   Пример метаданных первого фрагмента: {chunked_documents.metadata}")
else:
    print("   Не удалось создать фрагменты из документов.")
    exit()

# --- 3. Генерация эмбеддингов: Преобразование текста в векторы ---
# Используем популярную и легкую модель для локального развертывания
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
print(f"Шаг 3: Модель эмбеддингов '{embedding_model_name}' успешно загружена.")

# --- 4. Настройка и загрузка векторной базы данных: Локальное хранение эмбеддингов ---
# Определяем путь для сохранения базы данных Chroma [28, 29, 24, 13, 17, 18]
persist_directory = "./chroma_db"

# Создаем векторное хранилище из фрагментов документов с использованием выбранной модели эмбеддингов [2, 28, 19, 14, 6, 17, 18, 30]
# Если база данных уже существует, она будет загружена, иначе будет создана новая.
print(f"Шаг 4: Создание/загрузка векторной базы данных ChromaDB в '{persist_directory}'...")
vectorstore = Chroma.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
    persist_directory=persist_directory
)

# Chroma automatically persists when using persist_directory
print(f"   Векторная база данных ChromaDB создана и сохранена.")

# Преобразуем векторное хранилище в ретривер для использования в конвейере RAG [9, 31, 28, 5, 19, 14, 6, 18, 30]
retriever = vectorstore.as_retriever(k=4) # Извлекаем 4 наиболее релевантных документа
print("   Ретривер успешно инициализирован.")

# --- 5. Настройка генеративного компонента: Интеграция LLM ---

# ВЫБЕРИТЕ ОДИН ИЗ ВАРИАНТОВ НИЖЕ, РАСКОММЕНТИРУЙТЕ ЕГО И ЗАКОММЕНТИРУЙТЕ ОСТАЛЬНЫЕ

# --- ВАРИАНТ 1: Локальная LLM через Ollama (по умолчанию) ---
local_llm_model_name = "llama3.1" # Убедитесь, что эта модель загружена в Ollama
llm = OllamaLLM(model=local_llm_model_name)
print(f"Шаг 5: Локальная LLM '{local_llm_model_name}' успешно инициализирована через Ollama.")

# --- ВАРИАНТ 2: Глобальная LLM через OpenAI ---
# from langchain_openai import ChatOpenAI
# if "OPENAI_API_KEY" not in os.environ:
#     print("Ошибка: Переменная среды OPENAI_API_KEY не установлена.")
#     print("Пожалуйста, установите ее перед запуском скрипта.")
#     exit()
# global_llm_model_name = "gpt-3.5-turbo" # Или "gpt-4o"
# llm = ChatOpenAI(model=global_llm_model_name, temperature=0.7)
# print(f"Шаг 5: Глобальная LLM '{global_llm_model_name}' успешно инициализирована через OpenAI.")

# --- ВАРИАНТ 3: Глобальная LLM через Google Gemini ---
# from langchain_google_genai import ChatGoogleGenerativeAI
# if "GOOGLE_API_KEY" not in os.environ:
#     print("Ошибка: Переменная среды GOOGLE_API_KEY не установлена.")
#     print("Пожалуйста, установите ее перед запуском скрипта.")
#     exit()
# global_llm_model_name = "gemini-pro" # Или "gemini-1.5-flash"
# llm = ChatGoogleGenerativeAI(model=global_llm_model_name, temperature=0.7)
# print(f"Шаг 5: Глобальная LLM '{global_llm_model_name}' успешно инициализирована через Google Gemini.")

# --- ВАРИАНТ 4: Глобальная LLM через Anthropic Claude ---
# from langchain_anthropic import ChatAnthropic
# if "ANTHROPIC_API_KEY" not in os.environ:
#     print("Ошибка: Переменная среды ANTHROPIC_API_KEY не установлена.")
#     print("Пожалуйста, установите ее перед запуском скрипта.")
#     exit()
# global_llm_model_name = "claude-3-sonnet-20240229" # Или "claude-3-opus-20240229"
# llm = ChatAnthropic(model=global_llm_model_name, temperature=0.7)
# print(f"Шаг 5: Глобальная LLM '{global_llm_model_name}' успешно инициализирована через Anthropic.")


# --- 6. Композиция цепочки RAG: Сборка сквозной системы ---
# Шаблон запроса инструктирует LLM, как использовать извлеченный контекст.
# {context} будет заменен извлеченными фрагментами, а {question} - запросом пользователя.
prompt_template = """Используйте следующий контекст для ответа на вопрос.
Если вы не знаете ответа, просто скажите, что не знаете, не пытайтесь его выдумать.
Сохраняйте ответ кратким и точным.

Контекст:
{context}

Вопрос: {question}

Ответ:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
print("Шаг 6: Шаблон запроса для LLM создан.")

# Используем LangChain Expression Language (LCEL) для создания гибкой цепочки. [32]
# 1. Извлечь документы по запросу.
# 2. Передать извлеченные документы в качестве контекста и исходный вопрос в шаблон запроса.
# 3. Передать заполненный запрос в LLM.
# 4. Разобрать вывод LLM в строку.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
)
print("   Цепочка RAG успешно скомпонована.")

# --- 7. Выполнение запросов ---
print("\n--- Готово к выполнению запросам ---")

while True:
    user_query = input("\nВведите ваш вопрос (или 'выход' для завершения): ")
    if user_query.lower() == 'выход':
        print("Завершение работы RAG-системы.")
        break

    print(f"\nЗапрос пользователя: {user_query}")

    # Выполняем цепочку RAG
    try:
        response = rag_chain.invoke(user_query)
        print(f"\nОтвет RAG-системы:\n{response}")
    except Exception as e:
        print(f"Произошла ошибка при выполнении запроса: {e}")
        print("Убедитесь, что Ollama запущен и модель LLM ('llama3.1' или выбранная вами) доступна.")

8. Запуск приложения
Запустите Ollama с моделью: Откройте первый терминал и выполните:
Bash

ollama run llama3.1
Оставьте этот терминал открытым.
Запустите Python-скрипт: Откройте второй терминал, перейдите в директорию проекта, активируйте виртуальное окружение и выполните:
Bash

python rag_app.py
При первом запуске будет создана папка chroma_db/ и проиндексированы ваши статьи. Последующие запуски будут использовать существующую базу данных.
9. Использование
После запуска rag_app.py вы увидите приглашение ввести свой вопрос. Введите вопрос, связанный с содержимым ваших статей, и система сгенерирует ответ.

Примеры вопросов:

"Что такое Retrieval-Augmented Generation?"
"Какие преимущества у RAG-систем?"
"Расскажите о важности разбиения текста на фрагменты."
"Кто такой доктор Эмили Браун?" (если это имя есть в ваших статьях)
Чтобы завершить работу приложения, введите выход и нажмите Enter.

10. 🛠️ Устранение неполадок
ollama: command not found: См. раздел "3.2. Ollama" выше.
'list' object has no attribute 'page_content': Эта ошибка указывает на то, что вы пытаетесь получить доступ к атрибуту page_content у списка, а не у отдельного объекта Document. Убедитесь, что вы используете articles_documents.page_content и chunked_documents.page_content для доступа к первому элементу списка Document при выводе примеров. Код в rag_app.py уже исправлен.
Expected the jq schema to result in a list of objects (dict): Убедитесь, что jq_schema установлен на '.[ ]' в JSONLoader, чтобы он правильно итерировал по объектам в корневом массиве JSON.  Код в rag_app.py уже исправлен.   
Ошибки подключения к LLM (например, "Connection refused"):
Убедитесь, что Ollama запущен в отдельном терминале.
Убедитесь, что вы загрузили модель LLM (например, llama3.1) с помощью ollama run llama3.1.
Проверьте, что Ollama работает на порту по умолчанию 11434 (можно проверить в браузере по адресу http://127.0.0.1:11434).
Ошибки, связанные с памятью (MemoryError): Если вы используете большую LLM, возможно, у вас недостаточно оперативной памяти или VRAM на GPU.  Попробуйте использовать меньшую модель LLM (например, llama3.1:8b или другую, более компактную модель, доступную в Ollama).    
Проблемы с API-ключами (для глобальных LLM): Убедитесь, что вы правильно установили переменную среды с вашим API-ключом (OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY) перед запуском скрипта.
11. 📈 Расширение и дальнейшее развитие
Оптимизация фрагментов: Экспериментируйте с chunk_size и chunk_overlap. Рассмотрите более продвинутые стратегии разбиения на фрагменты, такие как семантическое разбиение, если ваши данные очень сложны.    
Выбор модели эмбеддингов: Оцените другие открытые модели эмбеддингов (например, из семейства BAAI bge или Mixedbread) для вашей предметной области.    
Расширенные стратегии извлечения:
Переранжирование (Reranking): Используйте модели переранжирования (например, BAAI bge-reranker-v2-m3 ) для улучшения релевантности извлеченных фрагментов после первоначального векторного поиска.    
Фильтрация по метаданным: Используйте метаданные (например, publication_date, category) для сужения области поиска.    
Масштабируемость: Для очень больших наборов данных рассмотрите переход на более масштабируемые векторные базы данных, такие как Milvus, Qdrant или Pinecone.    
Пользовательский интерфейс: Интегрируйте систему с веб-интерфейсом (например, Streamlit, FastAPI) для более удобного взаимодействия.    
Обработка ошибок и мониторинг: Для производственных внедрений добавьте более надежную обработку ошибок, логирование и мониторинг производительности. 