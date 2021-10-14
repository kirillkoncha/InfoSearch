import numpy as np
import os
from scipy import sparse


curr_dir = os.getcwd()
matrixesCorpus = []
matrixesTopics = []


for file in sorted(os.listdir(os.path.join(curr_dir, 'matrixes'))):
    if file.startswith('corpus'):
        matrixesCorpus.append(file)
    if file.startswith('topics'):
        matrixesTopics.append(file)


def get_cosine_similarity(sparse_matrix, query):
    return np.dot(sparse_matrix, query.T).toarray()


def get_scores(matrixesCorpus, matrixesTopics):
    names = [matrix[6:-4] for matrix in matrixesCorpus]
    models_scores = dict(zip(names, [0 for x in range(0, len(names))]))
    for i, model in enumerate(matrixesCorpus):
        score = 0
        sparse_matrix = sparse.load_npz(os.path.join(curr_dir, 'matrixes', model))
        queries = sparse.load_npz(os.path.join(curr_dir, 'matrixes', matrixesTopics[i]))
        res_mat = get_cosine_similarity(sparse_matrix, queries)
        sorted = np.argsort(-res_mat, axis=1)

        for index, row in enumerate(sorted):
            top_results = row[:10]
            if index in top_results:
                score += 1



        score = score/len(sorted)
        models_scores[model[6:-4]] = score

    return models_scores


print(get_scores(matrixesCorpus, matrixesTopics))


