import json
import numpy as np

from scipy import sparse
from pymystem3 import Mystem
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

m = Mystem()
tokenizer = RegexpTokenizer(r'\w+')
stopword = stopwords.words('russian')
count_vectorizer = CountVectorizer()
tf_vectorizer = TfidfVectorizer(use_idf=False)
tfidf_vectorizer = TfidfVectorizer(use_idf=True)


def get_corpus(filename):
    texts = []

    with open(filename) as json_file:
        corpus = list(json_file)[:50000]
    for i in range(0, 50000):
        issue = json.loads(corpus[i])
        answers = issue['answers']
        max_rating = 0
        idx = 0
        if len(answers) > 0:
            for index, answer in enumerate(answers):
                value = answer['author_rating']['value']
                if len(value) > 0 and int(value) > max_rating:
                    idx = index
                    max_rating = int(value)

            texts.append(answers[idx]['text'])

    texts = np.array(texts)

    return texts


def get_corpus_index(texts):
    x_count_vec = count_vectorizer.fit_transform(texts)
    x_tf_vec = tf_vectorizer.fit_transform(texts)
    tfidf_vectorizer.fit_transform(texts)

    idf = tfidf_vectorizer.idf_
    idf = np.expand_dims(idf, axis=0)
    tf = x_tf_vec

    k = 2
    b = 0.75

    values = []
    rows = []
    cols = []

    len_d = x_count_vec.sum(axis=1)
    avdl = len_d.mean()
    B_1 = (k * (1 - b + b * len_d / avdl))
    B_1 = np.expand_dims(B_1, axis=-1)

    for i, j in zip(*tf.nonzero()):
        rows.append(i)
        cols.append(j)
        A = tf[i, j] * idf[0][j] * (k + 1)
        B = tf[i, j] + B_1[i]
        value = A / B
        values.append(value[0][0])

    sparse_matrix = sparse.csr_matrix((values, (rows, cols)))

    return sparse_matrix


def get_cosine_similarity(sparse_matrix, query):

    return cosine_similarity(sparse_matrix, query)


def search_answer(sparse_matrix, query, corpus):
    scores = get_cosine_similarity(sparse_matrix, query)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    corpus = corpus[sorted_scores_indx.ravel()]

    return corpus


def preprocessing(texts):
    preprocessed_texts = []
    for text in texts:
        text_stripped = text.rstrip()
        text_stripped = ' '.join(tokenizer.tokenize(text_stripped.lower()))
        lemmas = m.lemmatize(text_stripped)
        lemmas = [w for w in lemmas if not w.isdigit() and w != ' ']
        preprocessed_texts.append(' '.join(lemmas))

    return preprocessed_texts


def get_search_query():
    query = input('Введите ваш запрос: ')
    query = preprocessing([query])

    return count_vectorizer.transform(query)


if __name__ == "__main__":
    corpus = get_corpus('questions_about_love.jsonl')
    prep_texts = preprocessing(corpus)
    sparse_matrix = get_corpus_index(prep_texts)
    while True:
        query = get_search_query()
        answers = search_answer(sparse_matrix, query, corpus)
        for i in range(0, 9):
                print(answers[i])
