# Simple-RAG
Простая RAG-система для файлов в .md формате

## Содержание

- [Структура проекта](#структурапроекта)
- [Использованные модели](#использованныемодели)
- [Установка и запуск](#установкаизапуск)



## Структура проекта

```sh
project/
├── embedding.py
├──  main.py
├──  semantic_chunker.py
```
- В файле embedding.py логика chunking и embedding. Перед запуском 
надо убедиться, что папки и .md файлы мастерской находятся в файле data.
Предусмотрена возможности сразу загрузить архив. 
- Во файле main.py скачивание модели gemma-7b-it и использование RAG.

## Использованные модели
Были использованы две модели:
1. gte-multilingual-base по ссылке https://huggingface.co/Alibaba-NLP/gte-multilingual-base.
2. google/gemma-7b-it по ссылке https://huggingface.co/google/gemma-7b-it.

Для скачивания второй модели понадобится токен HuggingFace и принять условия пользования по ссылке
    

## Установка и запуск
1. Установите зависемости
```sh
pip install -qU langchain_huggingface transformers bitsandbytes
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -qU langchain_community langchain_core
pip install faiss-cpu
```
2. Создайте папку data и разместите в ней нужные материалы. 



