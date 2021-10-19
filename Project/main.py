import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn
import nltk
import itertools
import os
import scipy
import time
import torch
import pymorphy2
import re


from gensim.models import KeyedVectors
from nltk.tokenize import RegexpTokenizer
from pymystem3 import Mystem
from scipy import sparse
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import RegexpTokenizer

curr_dir = os.getcwd()
morph = pymorphy2.MorphAnalyzer()

@st.cache()
def get_matrixes():
    count_matrix = scipy.sparse.load_npz(os.path.join(curr_dir, 'matrixes', 'topicsCount.npz'))
    tfidf_matrix = scipy.sparse.load_npz(os.path.join(curr_dir, 'matrixes', 'topicsTfIdf.npz'))
    BM25_matrix = scipy.sparse.load_npz(os.path.join(curr_dir, 'matrixes', 'topicsBM25.npz'))
    fasttext_matrix = scipy.sparse.load_npz(os.path.join(curr_dir, 'matrixes', 'topicsFasttext.npz'))
    bert_matrix = scipy.sparse.load_npz(os.path.join(curr_dir, 'matrixes', 'topicsBert.npz'))

    return count_matrix, tfidf_matrix, BM25_matrix, fasttext_matrix, bert_matrix


@st.cache(allow_output_mutation=True)
def get_vectorizers():
    count = open(os.path.join(curr_dir, 'vectorizers', 'tf_vectorizer.pickle'), 'rb')
    count = pickle.load(count)
    tfidf = open(os.path.join(curr_dir, 'vectorizers', 'tfidf_vectorizer.pickle'), 'rb')
    tfidf = pickle.load(tfidf)
    bm25 = open(os.path.join(curr_dir, 'vectorizers', 'tf_vectorizerBM25.pickle'), 'rb')
    bm25 = pickle.load(bm25)

    return count, tfidf, bm25


@st.cache(allow_output_mutation=True)
def get_other():
    tokenizer = RegexpTokenizer(r'\w+')
    m = Mystem()
    return tokenizer, m


@st.cache(allow_output_mutation=True)
def get_models():
    model_file = '/Users/kirillkonca/Downloads/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
    model = KeyedVectors.load(model_file)
    b_tokenizer = AutoTokenizer.from_pretrained('/Users/kirillkonca/PycharmProjects/streamlit/rubert-tiny')
    b_model = AutoModel.from_pretrained('/Users/kirillkonca/PycharmProjects/streamlit/rubert-tiny')
    return model, b_tokenizer, b_model


@st.cache(allow_output_mutation=True)
def get_data():
    top_ten = open('top_ten.pkl', 'rb')
    top_ten = pickle.load(top_ten)
    low_ten = open('not_top_ten.pkl', 'rb')
    low_ten = pickle.load(low_ten)
    nltk.download('stopwords')
    stopword = stopwords.words('russian')
    with open("corpus.txt", 'r') as f:
        corpus = [line.rstrip('\n') for line in f]
    corpus = np.array(corpus)
    return top_ten, low_ten, stopword, corpus


sparse_matrix = sparse.csr_matrix([])
top_ten, low_ten, stopword, corpus = get_data()
model, b_tokenizer, b_model = get_models()
tokenizer, m = get_other()
count, tfidf, bm25 = get_vectorizers()
count_matrix, tfidf_matrix, BM25_matrix, fasttext_matrix, bert_matrix = get_matrixes()


def get_query_bert(query, model, tokenizer):
    query = [query]
    vectors = []
    for text in query:
        t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        vectors.append(embeddings[0].cpu().numpy())

    return sparse.csr_matrix(vectors)


def get_query_fasttext(query):
    vectors = []
    for text in query:
        tokens = text.split()
        tokens_vectors = np.zeros((len(tokens), model.vector_size))
        vec = np.zeros((model.vector_size,))
        for idx, token in enumerate(tokens):
            tokens_vectors[idx] = model[token]
        if tokens_vectors.shape[0] != 0:
            vec = np.mean(tokens_vectors, axis=0)
            vec = normalize_vec(vec)
        vectors.append(vec)

    corpus = sparse.csr_matrix(vectors)

    return corpus


def get_query_tfidf(query, vectorizer):
    return vectorizer.transform(query)


def get_query_BM25(query, vectorizer):
    return vectorizer.transform(query)


def get_query_count(query, vectorizer):
    return vectorizer.transform(query)


def preprocessing(texts):
    preprocessed_texts = []
    for text in texts:
        text_stripped = text.rstrip()
        text_stripped = ' '.join(tokenizer.tokenize(text_stripped.lower()))
        lemmas = m.lemmatize(text_stripped)
        lemmas = [w for w in lemmas if not w.isdigit() and w != ' ' and w not in stopword]
        preprocessed_texts.append(' '.join(lemmas))

    return np.array(preprocessed_texts)


@st.cache
def load_matrix(name):
    sparse_matrix = scipy.sparse.load_npz(os.path.join(curr_dir, 'matrixes', name))
    return sparse_matrix


def normalize_vec(x):
    return x / np.linalg.norm(x)


def get_cosine_similarity(sparse_matrix, query):
    return np.dot(sparse_matrix, query.T).toarray()


def search_answer(sparse_matrix, query, corpus, option):
    if option == 'Count Vectors':
        sparse_matrix = count_matrix
        vectorizer = count
        query = get_query_count(query, vectorizer)
    if option == 'TfIdf':
        sparse_matrix = tfidf_matrix
        vectorizer = tfidf
        query = get_query_tfidf(query, vectorizer)
    if option == 'BM25':
        sparse_matrix = BM25_matrix
        vectorizer = bm25
        query = get_query_BM25(query, vectorizer)
    if option == 'Fasttext':
        sparse_matrix = fasttext_matrix
        query = get_query_fasttext(query)
    if option == 'Bert':
        sparse_matrix = bert_matrix
        query = get_query_bert(query, b_model, b_tokenizer)
    scores = get_cosine_similarity(sparse_matrix, query)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    corpus = corpus[sorted_scores_indx.ravel()]

    return corpus


st.title('Ответы о любви ❤️')
st.markdown(
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
    unsafe_allow_html=True,
)

query_params = st.experimental_get_query_params()
tabs = ["Поиск", "Данные"]

len_data = pd.DataFrame(
    [[134403, 0], [0, 83444]],
    columns=['До препроца', 'После препроца']
)

tokens_data = pd.DataFrame(
    [[13976, 0], [0, 13046]],
    columns=['До препроца', 'После препроца']
)

if "tab" in query_params:
    active_tab = query_params["tab"][0]
else:
    active_tab = "Поиск"

if active_tab not in tabs:
    st.experimental_set_query_params(tab="Данные")
    active_tab = "Данные"

li_items = "".join(
    f"""
    <li class="nav-item">
        <a class="nav-link{' active' if t == active_tab else ''}" href="/?tab={t}">{t}</a>
    </li>
    """
    for t in tabs
)
tabs_html = f"""
    <ul class="nav nav-tabs">
    {li_items}
    </ul>
"""

st.markdown(tabs_html, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if __name__ == "__main__":
    if active_tab == "Данные":
        link = 'Поиск по [датасету](https://www.dropbox.com/s/6uinrxp7p33ha1u/questions_about_love.jsonl?dl=0) вопросов и ответов о любви с Ответов Mail.Ru.'
        st.markdown(link, unsafe_allow_html=True)
        st.write('Поиск осуществляется по первым 10К вопросов.')
        values = st.slider(
            'N самых частотных и низкочастотных слов в корпусе (справа абсолютная частотность)',
            1, 10)
        st.write('Самые частотные')
        st.write(dict(itertools.islice(top_ten.items(), values)))
        st.write('Самые низкочастотные')
        st.write(dict(itertools.islice(low_ten.items(), values)))
        st.write('Длина корпуса до и после препроца')
        st.bar_chart(len_data)
        st.write('Уникальные токены до и после препроца')
        st.bar_chart(tokens_data)
    elif active_tab == "Поиск":
        search = st.text_input("Введите ваш запрос", "")
        option = st.selectbox('Выберите способ поиска',
                              ('Count Vectors', 'TfIdf', 'BM25',
                               'Fasttext', 'Bert'))
        number = st.number_input('Введите число ответов: ', value=10, min_value=1, max_value=20, step=1)
        if st.button('Искать'):
            start = time.time()
            if option != 'Bert':
                search = preprocessing([search])
            st.subheader('{} лучших результатов:'.format(number))
            answers = search_answer(sparse_matrix, search, corpus, option)
            for i in range(0, number):
                st.write(answers[i])
            st.write('Время выполнения поиска: ', time.time() - start)

    else:
        st.error("Что-то пошло не так...")
