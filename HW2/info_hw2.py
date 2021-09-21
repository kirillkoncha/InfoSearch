import os


from pymystem3 import Mystem
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

vectorizer = TfidfVectorizer()
m = Mystem()
tokenizer = RegexpTokenizer(r'\w+')
stopword = stopwords.words('russian')
stopword.append('сезон')  # добавим в стоп-слова сезон, серию и автора субтитров
stopword.append('серия')
stopword.append('dais')


def preprocessing(path_file: str) -> list:
    """
    Препроцессинг: токенизация, лемматизация, удаление стоп-слов.
    Последние три строчки удаляются, потому что там информация по длине серии и
    другие техн.детали.
    """
    with open(path_file, 'r+', encoding='utf-8-sig') as file:
        lines = file.readlines()
        lines = lines[:-3]
        lines = [line.strip() for line in lines]
        lines = [tokenizer.tokenize(line.lower()) for line in lines
                 if line != '']
        for i in range(0, len(lines)):
            lines[i] = ' '.join(lines[i])
            lines[i] = m.lemmatize(lines[i])
            lines[i] = [word for word in lines[i] if word not in stopword
                        and not word.isdigit()]
            lines[i] = ' '.join(lines[i])
        line = ' '.join(lines)

    return line


def reading_and_indexing() -> tuple:
    """
    Чтение всех файлов. Папка с субтатрами должна лежать в папке с кодом.
    Называться 'friends-data'.
    Возвращает Term-Document матрицу.
    """
    corpus = []
    files = []
    friends_dir = os.path.join(os.getcwd(), 'friends-data')
    folders = os.listdir(friends_dir)
    for folder in folders:
        curr_dir = os.path.join(friends_dir, folder)
        episodes = os.listdir(curr_dir)
        for episode in episodes:
            files.append(episode)
            episode_file = os.path.join(curr_dir, episode)
            preprocessed_text = preprocessing(episode_file)
            corpus.append(preprocessed_text)
    bag_of_words = vectorizer.fit_transform(corpus)

    return bag_of_words.toarray(), files


def request(request_str: str) -> list:
    """
    Функция препроцессинга и трансформа запроса
    """
    request_str = ' '.join(tokenizer.tokenize(request_str.lower()))
    request_str = m.lemmatize(request_str)
    request_str = [word for word in request_str if word not in stopword
                   and not word.isdigit()]
    request_str = ' '.join(request_str)
    request_matrix = vectorizer.transform([request_str])

    return request_matrix.toarray()


def get_cosine_similarity(corpus: list, request: list) -> list:
    """
    Функция рассчета косинусной близости
    """
    cosine = cosine_similarity(corpus, request)

    return cosine


def main():
    """
    Главная функция, которая все объединяет 
    """
    index = reading_and_indexing()
    bag_of_words = index[0]
    corpus = index[1]
    while True:
        req = input("Введите запрос: ")
        sim = get_cosine_similarity(bag_of_words, request(req))
        answer = sorted(range(len(sim.flatten())), key=lambda k: sim.flatten()[k], reverse=True)
        for item in answer:
            print(corpus[item])


if __name__ == '__main__':
    main()

