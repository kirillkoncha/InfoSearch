import os
import nltk

from pymystem3 import Mystem
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


nltk.download('stopwords')
vectorizer = CountVectorizer(analyzer='word')
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


def reading_and_indexing() -> classmethod:
    """
    Чтение всех файлов. Папка с субтатрами должна лежать в папке с кодом.
    Называться 'friends-data'.
    Возвращает Term-Document матрицу.
    """
    corpus = []
    friends_dir = os.path.join(os.getcwd(), 'friends-data')
    folders = os.listdir(friends_dir)
    for folder in folders:
        curr_dir = os.path.join(friends_dir, folder)
        episodes = os.listdir(curr_dir)
        for episode in episodes:
            episode_file = os.path.join(curr_dir, episode)
            preprocessed_text = preprocessing(episode_file)
            corpus.append(preprocessed_text)
    bag_of_words = vectorizer.fit_transform(corpus)

    return bag_of_words


def top_words(bag_of_words: str, n=5) -> list:
    """
    Функция, чтобы посмотреть n самых частотных и редких слов.
    """
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx
                  in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return words_freq[:n] + words_freq[-n:]


def words_all_documents(bag_of_words: classmethod) -> list:
    """
    Считает слова, которые появляются во всех документах.
    Документов всего 165, если слово встречается в одном,
    то в словарь записывается 1.
    Если слово набирает 165, то значит, что
    оно встречается во всех документах.
    """
    bag_of_words = bag_of_words.toarray()
    word_occurrences = dict.fromkeys(range(0, len(bag_of_words[0])), 0)
    words = vectorizer.get_feature_names()
    for doc in bag_of_words:
        for i in range(0, len(doc)):
            if doc[i] != 0:
                word_occurrences[i] += 1

    for i in range(0, len(words)):
        word_occurrences[words[i]] = word_occurrences.pop(i)

    word_occurrences = dict(sorted(word_occurrences.items(),
                                   key=lambda item: item[1]))

    return [k for k, v in word_occurrences.items() if v == 165]


def popularity(bag_of_words: classmethod) -> dict:
    """
    Вывод имен персонажей и количество раз, которые они встречаются
    """
    all_names = ['джоуя', 'чендлер', 'рейчел', 'рейчэл', 'рэйч', 'рейч',
                 'фиби', 'росс', 'моника', 'мон', 'чен', 'чэндлер',
                 'фибс', 'джо', 'чэн']
    united_names = dict.fromkeys(['Джо', 'Моника', 'Рэйчел',
                                  'Чендлер', 'Фиби', 'Росс'], 0)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx
                  in vectorizer.vocabulary_.items()]
    names_freq = [word for word in words_freq if word[0]
                  in all_names]
    for name in names_freq:
        if name[0] in ['джоуя', 'джо']:
            united_names['Джо'] += name[1]
        if name[0] in ['чендлер', 'чен', 'чэндлер', 'чэн']:
            united_names['Чендлер'] += name[1]
        if name[0] in ['фиби', 'фибс']:
            united_names['Фиби'] += name[1]
        if name[0] in ['моника', 'мон']:
            united_names['Моника'] += name[1]
        if name[0] in ['росс']:
            united_names['Росс'] += name[1]
        if name[0] in ['рэйчел', 'рейчел', 'рейч', 'рэйч']:
            united_names['Рэйчел'] += name[1]

    return dict(sorted(united_names.items(), key=lambda item: item[1], reverse=True))


if __name__ == '__main__':
    bag_of_words = reading_and_indexing()
    print('Самое частотное и редкое слово:')
    print(top_words(bag_of_words, 1))
    print('Слова, которые есть во всех документах:')
    print(words_all_documents(bag_of_words))
    print('Количество упоминаний персонажей (видно, что самый частотный -- Росс): ')
    print(popularity(bag_of_words))
    
