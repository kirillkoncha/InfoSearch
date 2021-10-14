import json
import numpy as np
import pickle
import os
import nltk


from scipy import sparse
from pymystem3 import Mystem
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel


b_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
b_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")


nltk.download('stopwords')
stopword = stopwords.words('russian')
curr_dir = os.getcwd()
model_file = '/Users/kirillkonca/Downloads/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
model = KeyedVectors.load(model_file)
m = Mystem()
tokenizer = RegexpTokenizer(r'\w+')
count_vectorizer = CountVectorizer()
tf_vectorizer = TfidfVectorizer(use_idf=False)
tfidf_vectorizer = TfidfVectorizer(use_idf=True)


def get_bert_corpus(texts, model, tokenizer):
    vectors = []
    for text in texts:
        t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        vectors.append(embeddings[0].cpu().numpy())

    return sparse.csr_matrix(vectors)


def get_corpus(filename):
    texts = []
    topics = []
    with open(filename, encoding='utf-8') as json_file:
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
            topic = ' '.join([issue['question'], issue['comment']])
            topics.append(topic)
            texts.append(answers[idx]['text'])

    return np.array(texts), np.array(topics)


def get_fasttext_corpus(prep_texts):
    vectors = []
    for text in prep_texts:
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


def normalize_vec(x):
    return x / np.linalg.norm(x)


def get_BM25(texts):
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


def get_tfidf_corpus(texts):
    return tfidf_vectorizer.fit_transform(texts)


def get_count_vect_corpus(texts):
    '''
    Тут tf_vectorizer, потому что без
    idf=true он такой же, как и count vectors
    но с нормализацией
    '''
    return tf_vectorizer.fit_transform(texts)


def get_cosine_similarity(sparse_matrix, query):
    return np.dot(sparse_matrix, query.T).toarray()


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
        lemmas = [w for w in lemmas if not w.isdigit() and w != ' ' and w not in stopword]
        preprocessed_texts.append(' '.join(lemmas))

    return np.array(preprocessed_texts)


def get_query_count():
    query = input('Введите ваш запрос: ')
    query = preprocessing([query])
    return tf_vectorizer.transform(query)


def get_query_fasttext():
    query = input('Введите ваш запрос: ')
    query_matrix = get_fasttext_corpus([query])
    return query_matrix


def get_query_BM25():
    query = input('Введите ваш запрос: ')
    query = preprocessing([query])
    return count_vectorizer.transform(query)


def get_query_tfidf():
    query = input('Введите ваш запрос: ')
    query = preprocessing([query])

    return tfidf_vectorizer.transform(query)


def get_query_bert():
    query = input('Введите ваш запрос: ')
    cls_embeddings = get_bert_corpus([query], b_model, b_tokenizer)
    return sparse.csr_matrix(cls_embeddings)


if __name__ == "__main__":
    corpus, topics = get_corpus('questions_about_love.jsonl')
    method = input('Выберите метод поиска 1. Count Vectorizer,'
                   '2. TfIdf Vectorizer,'
                   '3. BM25,'
                   '4. fasttext,'
                   '5. bert: ')
    if method == '1':
        sparse_matrix = sparse.load_npz(os.path.join(curr_dir, 'matrixes', 'corpusCount.npz'))
        tf_vectorizer = pickle.load(open(os.path.join(curr_dir, 'vectorizers', 'tf_vectorizer.pickle'), "rb"))
        get_query = get_query_count
    if method == '2':
        sparse_matrix = sparse.load_npz(os.path.join(curr_dir, 'matrixes', 'corpusTfIdf.npz'))
        tfidf_vectorizer = pickle.load(open(os.path.join(curr_dir, 'vectorizers', 'tfidf_vectorizer.pickle'), "rb"))
        get_query = get_query_tfidf
    if method == '3':
        sparse_matrix = sparse.load_npz(os.path.join(curr_dir, 'matrixes', 'corpusBM25.npz'))
        count_vectorizer = pickle.load(open(os.path.join(curr_dir, 'vectorizers', 'tf_vectorizerBM25.pickle'), "rb"))
        get_query = get_query_BM25
    if method == '4':
        sparse_matrix = sparse.load_npz(os.path.join(curr_dir, 'matrixes', 'corpusFasttext.npz'))
        get_query = get_query_fasttext
    if method == '5':
       sparse_matrix = sparse.load_npz(os.path.join(curr_dir, 'matrixes', 'corpusBert.npz'))
       get_query = get_query_bert
    while True:
        query = get_query()
        answers = search_answer(sparse_matrix, query, corpus)
        # Топ 5 ответов
        for i in range(0, 4):
            print(answers[i])


