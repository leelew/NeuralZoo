from sklearn import neighbors

class knn():

    def __init__(self, n_neighbors=5):

        self.n_neighbors = n_neighbors


    def fit(self, X, y):

        self.regressor = neighbors.KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
        )

        self.regressor.fit(X, y)

        return self