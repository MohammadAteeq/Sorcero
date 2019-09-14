import numpy as np


class SimilarityComputation:

    def inner_product(self, matrix_a, matrix_b):
        return np.inner(matrix_a, matrix_b)

    def compute_cosine(self, va, vb):
        return np.dot(va, vb) / (np.sqrt(np.dot(va, va)) * np.sqrt(np.dot(vb, vb)))
