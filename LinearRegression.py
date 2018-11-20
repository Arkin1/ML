class Linear_Regression:
  
    def __init__(self, learning_rate, absolute_error):
        self.alpha = learning_rate
        self.err = absolute_error                 

    def fit(self, X, Y):
           m = X.shape[0]
           X = np.append(X, np.ones((m,1)), axis=1)

           X = np.roll(X,1,1)

           n = X.shape[1]

           self.coef  = np.random.randint(n, size=(n,1)).astype('float64')
      
           X = np.matrix(X)
           Y = np.matrix(Y)
           Y = Y.T
           self.coef = np.matrix(self.coef)
   
           last_cost = 0
           current_cost = 100

           while(math.fabs(current_cost - last_cost) > self.err):
                 last_cost = current_cost
                 self.coef = self.coef - self.alpha/m*(((X*self.coef - Y)).T*X).T
             
                 current_cost = np.power((X*self.coef - Y),2).sum()/(2*m)

    def predict(self, X):
        X = np.append(X, np.ones((X.shape[0],1)), axis=1)
        X = np.roll(X,1,1)
        return (X*self.coef)
