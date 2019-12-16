from math import sqrt
from Similarity import Similarity

class CosineSimilarity(Similarity):
    
    def __init__(self, coorA = None, coorB = None):
        super().__init__(coorA, coorB)

    def calculate(self):
        ab, a, b = 0, 0, 0
        for i in range(0, len(self.coorA)):
            ab += self.coorA[i] * self.coorB[i]
            a += self.coorA[i]**2
            b += self.coorB[i]**2
        self.distance = ab/(sqrt(a)*sqrt(b))