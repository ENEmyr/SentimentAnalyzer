from math import sqrt

# Euclidean distance in default
class Similarity:
    distance = 0
    coorA = None
    coorB = None
    
    def __init__(self, coorA = None, coorB = None):
        if coorA and coorB:
            if len(coorA) >= 2 and len(coorB) >= 2 and len(coorA) == len(coorB):
                self.coorA = coorA
                self.coorB = coorB
        
    def set_points(self, coorA, coorB):
        if len(coorA) >= 2 and len(coorB) >= 2 and len(coorA) == len(coorB):
            self.coorA = coorA
            self.coorB = coorB
            return self
        else:
            raise('Require a same size list of coordinates of A point and B point.')
    
    def calculate(self):
        for i in range(0, len(self.coorA)):
            self.distance += (self.coorA[i] - self.coorB[i])**2
        self.distance = sqrt(self.distance)

    def get_distance(self):
        if not(self.coorA and self.coorB):
            raise('Require a coordinates of 2 points in process to find the distance between it.')
        else:
            self.calculate()
        return self.distance
