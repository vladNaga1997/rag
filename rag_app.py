import json
from pathlib import Path

# Импорт компонентов LangChain
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document  # Для явного импорта Document

print("Начало настройки RAG-системы...")

# --- 1. Прием данных: Загрузка статей из локальных JSON-файлов ---
# Путь к вашему JSON-файлу со статьями
file_path = "./your_articles.json"


# Определяем функцию метаданных для извлечения конкретных полей из каждой записи статьи [1]
def extract_article_metadata(record: dict, metadata: dict) -> dict:
    metadata["article_id"] = record.get("article_id")
    metadata["title"] = record.get("title")
    metadata["author"] = record.get("author")
    metadata["publication_date"] = record.get("publication_date")
    metadata["category"] = record.get("category")
    return metadata


try:
    # Инициализируем JSONLoader для обработки каждого объекта статьи [1]
    # jq_schema '.[ ]' выбирает каждый объект в корневом массиве JSON
    # content_key указывает 'full_content' как основной текст для Document [1]
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[ ]",
        content_key="full_content",
        metadata_func=extract_article_metadata,
    )

    # Загружаем документы
    articles_documents = loader.load()
    print(
        f"Шаг 1: Загружено {len(articles_documents)} объектов LangChain Document из JSON-файла."
    )
    if articles_documents:
        print(
            f"   Пример содержимого первого документа (первые 100 символов): {articles_documents[0].page_content[:100]}..."
        )
        print(
            f"   Пример метаданных первого документа: {articles_documents[0].metadata}"
        )
    else:
        print(
            "   В JSON-файле не найдено документов для загрузки. Убедитесь, что 'your_articles.json' не пуст."
        )
        exit()  # Выходим, если нет документов для обработки
except FileNotFoundError:
    print(
        f"Ошибка: Файл {file_path} не найден. Убедитесь, что 'your_articles.json' находится в той же директории."
    )
    exit()
except json.JSONDecodeError:
    print(
        f"Ошибка: Не удалось декодировать JSON из {file_path}. Проверьте формат файла."
    )
    exit()
except Exception as e:
    print(f"Произошла ошибка при загрузке JSON: {e}")
    exit()

# --- 2. Предварительная обработка документов: Эффективное разбиение текста на фрагменты ---
# Инициализируем RecursiveCharacterTextSplitter [2, 3, 4]
# chunk_size определяет максимальный размер каждого фрагмента [3, 4]
# chunk_overlap определяет количество символов, которые будут перекрываться между соседними фрагментами [3, 4]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Оптимальный размер фрагмента зависит от LLM и данных
    chunk_overlap=200,  # Рекомендуется 10-20% перекрытия [5]
    separators=["\n\n", "\n", " ", ""],  # По умолчанию, но можно настроить [4]
)

# Разбиваем загруженные документы на фрагменты
chunked_documents = text_splitter.split_documents(articles_documents)
print(f"Шаг 2: Создано {len(chunked_documents)} фрагментов из исходных документов.")
if chunked_documents:
    print(
        f"   Пример содержимого первого фрагмента (первые 200 символов): {chunked_documents[0].page_content[:200]}..."
    )
    print(f"   Пример метаданных первого фрагмента: {chunked_documents[0].metadata}")
else:
    print("   Не удалось создать фрагменты из документов.")
    exit()

# --- 3. Генерация эмбеддингов: Преобразование текста в векторы ---
# Используем популярную и легкую модель для локального развертывания [6]
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
print(f"Шаг 3: Модель эмбеддингов '{embedding_model_name}' успешно загружена.")

# --- 4. Настройка и загрузка векторной базы данных: Локальное хранение эмбеддингов ---
# Определяем путь для сохранения базы данных Chroma [7, 8]
persist_directory = "./chroma_db"

# Создаем векторное хранилище из фрагментов документов с использованием выбранной модели эмбеддингов [9]
# Если база данных уже существует, она будет загружена, иначе будет создана новая.
print(
    f"Шаг 4: Создание/загрузка векторной базы данных ChromaDB в '{persist_directory}'..."
)
vectorstore = Chroma.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
    persist_directory=persist_directory,
)

# Chroma automatically persists when using persist_directory
print(f"   Векторная база данных ChromaDB создана и сохранена.")

# Преобразуем векторное хранилище в ретривер для использования в конвейере RAG [10, 11]
retriever = vectorstore.as_retriever(k=4)  # Извлекаем 4 наиболее релевантных документа
print("   Ретривер успешно инициализирован.")

# --- 5. Настройка генеративного компонента: Интеграция локальной LLM ---
# Убедитесь, что Ollama установлен и запущена модель (например, llama3.1) [12, 13]
# ollama run llama3.1
local_llm_model_name = "llama3.1"  # Или "gemma:7b", "mistral", и т.д.
llm = OllamaLLM(model=local_llm_model_name)
print(
    f"Шаг 5: Локальная LLM '{local_llm_model_name}' успешно инициализирована через Ollama."
)

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

# Используем LangChain Expression Language (LCEL) для создания гибкой цепочки. [14]
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
print("\n--- Готово к выполнению запросов ---")

while True:
    user_query = input("\nВведите ваш вопрос (или 'выход' для завершения): ")
    if user_query.lower() == "выход":
        print("Завершение работы RAG-системы.")
        break

    print(f"\nЗапрос пользователя: {user_query}")

    # Выполняем цепочку RAG
    try:
        response = rag_chain.invoke(user_query)
        print(f"\nОтвет RAG-системы:\n{response}")
    except Exception as e:
        print(f"Произошла ошибка при выполнении запроса: {e}")
        print(
            "Убедитесь, что Ollama запущен и модель LLM ('llama3.1' или выбранная вами) доступна."
        )
