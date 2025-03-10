import pandas as pd
import numpy as np


class LogisticRegression():
    def __init__(self):
        # Model Definition
        self.slopes = None
        self.intercept = None

        # Model Parameters
        self.method = None
        self.learning_rate = None
        self.epsilon = None
        self.penalty = None
        self.max_iters = None
        self.max_iters_no_change = None

        # Extra metrics
        self.errors = []
        self.iterations = []

    def __str__(self):
        return f'Logistic Regression Model\n' \
               f'Coefficients: \n{self.slopes}\n' \
               f'Method: {self.method}\n' \
               f'Learning Rate: {self.learning_rate}\n'

    def fit(self, X, y, **kwargs):
        """
           This method actually builds the model.

           :parameter x - Pandas DataFrame containing explanatory variables
           :parameter y - Pandas DataFrame containing response variables
           :parameter method - String that determines the method used to build the model.
            Default is Stochastic Gradient Descent. Must be one of: 'SGD', <others>
           """
        method = kwargs['method'] if 'method' in kwargs else 'SGD'

        # Feed in data one at a time
        if method == 'SGD':
            self.method = method
            self._fit_by_sgd(X, y, **kwargs)
        else:
            return
    def _fit_by_sgd(self,X, y, **kwargs):
        """
        Calculates the gradient of the loss function for linear regression.
        ∇L(x) = -2X^T(y - Xβ)

        :param x: Column vector of explanatory variables
        :param y: Column vector of dependent variables
        :param learning_rate: Rate that the gradient descent learns at
        :param epsilon: Bound for error. This determines when gradient descent will stop
        :param alpha: Coefficient used in penalty term for l1 and l2 regularization
        :param max_iters: Maximum number of iterations for gradient descent. Default is 1000
        :param max_iters_no_change: Determines how early we should stop if there is little change in error. Default is 10.
        :return: Vector of parameters for Linear Regression
        """
        assert len(X) == len(y)
        
        # Instantiate 1, 0,...,0 as betas where the one is the intercept
#         betas = np.concatenate(np.array([1,]), np.zeros(X.shape[1]))
        betas = np.random.rand(X.shape[1]+1)
        
        # Insert a 1 at beginning of each row for x's to represent intercept
        X.insert(0, '__intercept__', 1)
        X = X.to_numpy()
        y = y.to_numpy()
    
        def apply_beta(betas, data_row):
            # multiply betas by xs
            beta_by_xs = np.dot(betas, data_row)

            return 1 / (1 + np.e**(-1 *beta_by_xs))

        # Update Betas
        def update_via_gradient(rate, p, y, betas, X):
            # update betas one at a time
            for beta_index in range(len(betas)):
                betas[beta_index] = betas[beta_index] + (rate * (y - p) * X[beta_index]) 
            return betas

        def _fit(X, y, betas,rate = .1,epsilon = .001):

            current_Loss = 0
            prior_Loss = np.inf
            
            # Stop looping when regression fails to change by eps in either direction
            while abs(current_Loss -  prior_Loss) > epsilon:

                Loss = 0
                current_Loss = prior_Loss
                
                for row_index in range(X.shape[0]):
                    pt = X[row_index,:]
                    # Transform pt by applying betas
                    p = apply_beta(betas, pt)
                    # Update betas
                    betas = update_via_gradient(rate, p, y[row_index], betas,pt)
                    # Calculate difference between prediction and ground truth
                    Loss += abs(p - y[row_index])

                prior_Loss = Loss
                
            return betas
        
        # Fit betas and set slopes of model
        self.slopes = _fit(X, y, betas)
        
    def predict(self, x):
        """
        Makes predictions based on fit data. This means that fit() must be called before predict()

        :parameter x - Pandas DataFrame of data you want to make predictions about.
        """

        if self.slopes is None:
            print(f'Unable to make predictions until the model is fit. Please use fit() first.')
            return
        else:
            slopes = self.slopes
            
            x0 = x.to_numpy()
            
            beta_by_xs = [np.dot(slopes[1:], data_row) + slopes[0] for data_row in x0]

            return [1 / (1 + np.e**(-1 *beta_by_x)) for beta_by_x in beta_by_xs]

        
class LDA():
    def __init__(self):
        #LDA objects
        self.vector = None
        self.mu_1 = None
        self.mu_2 = None
        self.S = None
        self.B = None
        
    def get_stats(self,data, y):
        n_1 = 0
        n_2 = 0
        sum_1 = np.zeros(data.shape[1])
        sum_2 = np.zeros(data.shape[1])

        for (x_i, y_i) in zip(data, y):
            if y_i == 1:
                n_1 += 1
                sum_1 += x_i
            else:
                n_2 += 1
                sum_2 += x_i

        self.mu_1 = sum_1 / n_1
        self.mu_2 = sum_2 / n_2

        return
    def get_B(self,data, y):
        self.get_stats(data, y)

        self.B = np.matmul((self.mu_1 - self.mu_2).reshape(-1, 1), (self.mu_1 - self.mu_2).reshape(1, -1))
        return
    def get_S(self,data, y):
        self.get_stats(data, y)

        S_1 = np.zeros((data.shape[1], data.shape[1]))
        S_2 = np.zeros((data.shape[1], data.shape[1]))

        for (x_i, y_i) in zip(data, y):
            if y_i == 1:
                S_1 += np.matmul((x_i - self.mu_2).reshape(-1, 1), (x_i - self.mu_2).reshape(1, -1))
            else:
                S_2 += np.matmul((x_i - self.mu_2).reshape(-1, 1), (x_i - self.mu_2).reshape(1, -1))

        self.S = S_1 + S_2
        return
    
    def fit(self,data,y):
        self.get_stats(data,y)
        self.get_B(data,y)
        self.get_S(data,y)
        eig_values, eig_vectors = np.linalg.eig(np.matmul(np.linalg.inv(self.S), self.B))
        self.vector = np.real(eig_vectors[eig_values.argmax()])
        
    def predict(self,x):
        preds = []
        if self.vector == None:
            print(f'Unable to make predictions until the model is fit. Please use fit() first.') 
            return
        else:
            for x_i in x:
                proj = np.dot(self.vector,x_i)
                if abs(proj - self.mu_1) < abs(proj - self.mu_2):
                    preds.append(1)
                else:
                    preds.append(-1)
        return preds

