from Similarity import CosineSimilarity

if __name__ == '__main__':
    test = CosineSimilarity([1, 2, 3], [3, 5, 7])
    print(test.get_distance())