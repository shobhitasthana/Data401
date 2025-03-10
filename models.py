import pandas as pd
import numpy as np
import scipy

class LogReg():
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
        self.verbose = None
        self.lam = .5
        
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
    

        # Extra metrics
        self.errors = []
        self.iterations = []

    def __str__(self):
        return f'Logistic Regression Model\n' \
               f'Coefficients: \n{self.slopes}\n' \
               f'Method: {self.method}\n' \
               f'Learning Rate: {self.learning_rate}\n'

    def fit(self, X, y,**kwargs):
        """
           This method actually builds the model.

           :parameter x - Pandas DataFrame containing explanatory variables
           :parameter y - Pandas DataFrame containing response variables
           :parameter method - String that determines the method used to build the model.
            Default is Stochastic Gradient Descent. Must be one of: 'SGD', <others>
           """
        method = kwargs['method'] if 'method' in kwargs else 'SGD'
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False

        # Feed in data one at a time
        if method == 'SGD':
            self.method = method
            self._fit_by_sgd(X, y, **kwargs)
        else:
            return
    def _fit_by_sgd(self,X, y,**kwargs):
        """
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
        
        # Instantiate random betas
        betas = np.random.rand(X.shape[1]+1)
        
        # Insert a 1 at beginning of each row for x's to represent intercept
        X.insert(0, '__intercept__', 1)
        X = X.to_numpy()
        y = y.to_numpy()
        
        # Normalize X
        mins = np.min(X, axis = 0) 
        maxs = np.max(X, axis = 0) 
        rng = maxs - mins 
        X = 1 - ((maxs - X)/(rng + .000001))
    
        def apply_beta(betas, data_row):

            return 1 / (1 + np.e**(-1 * np.dot(betas, data_row)))

        # Update Betas
        def update_via_gradient(rate, p, y, betas, X):
            
            # update betas one at a time
            for beta_index in range(len(betas)):
                betas[beta_index] = betas[beta_index] + (rate * ((y - p) * X[beta_index]) - (rate*(self.lam * betas[beta_index]))) 
            return betas
        
        # Calculate loss for ridge regression
        def calc_loss(y, p, betas):
            return (-y * np.log(p + .000001))  - ((1 - y) *np.log(1-p + .0000001)) + (sum(betas) * self.lam)

        def _fit(X, y, betas,verbose, rate = .01,epsilon = .01):

            # Will be used as stopping condition
            prior_loss = np.inf
            curr_loss = 0
            max_epochs = 50
            epoch_count = 0
            
            # Stop looping when epoch loss fails to change by a total magnitude greater than epsilon
            while abs(prior_loss - curr_loss) > epsilon and epoch_count <= max_epochs:
                prior_loss = curr_loss
                curr_loss = 0
                for row_index in range(X.shape[0]):
                    pt = X[row_index,:]

                    # Transform pt by applying betas
                    p = apply_beta(betas, pt)
                    
                    curr_loss += calc_loss(y[row_index], p, betas)

                    # Update betas
                    betas = update_via_gradient(rate, p, y[row_index], betas,pt)
                epoch_count += 1
                                    

                
            return betas
        
        # Fit betas and set slopes of model
        self.slopes = _fit(X, y, betas, self.verbose)
        
    def predict(self, x, deterministic = True, thresh = .5):
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

            # Make a classification if deterministic == True
            if deterministic:
                return [1 if (1 / (1 + np.e**(-1 *beta_by_x))) >= thresh else 0 for beta_by_x in beta_by_xs]
            # Return Probability if deterministic == False
            else:
                return [1 / (1 + np.e**(-1 *beta_by_x)) for beta_by_x in beta_by_xs]
            
    def calc_accuracy(self, preds, true):
        self.accuracy = sum(preds == true) / true.shape[0]
        return self.accuracy
    
    def calc_precision(self, preds, true):
        true_positives = ((preds == 1) & (true == 1)).sum()
        self.precision = true_positives / (preds == 1).sum()
        return self.precision
    
    def calc_recall(self, preds, true):
        true_positives = ((preds == 1) & (true == 1)).sum()
        self.recall = true_positives / (true == 1).sum()
        return self.precision
    
    def calc_f1(self, preds, true):
        if self.precision == None:
            self.precision = self.calc_precision(preds, true)
        if self.recall == None:
            self.recall = self.calc_recall(preds, true)
        self.f1 = 2*(self.precision*self.recall) / (self.precision + self.recall)
        return self.f1
        
    
    def score(self, preds, true, metrics=['accuracy','precision','recall','f1']):
        scores = {}
        for metric in metrics:
            if metric == 'accuracy':
                scores['accuracy'] = self.calc_accuracy(preds, true)
            elif metric == 'precision':
                scores['precision'] = self.calc_precision(preds, true)
            elif metric == 'recall':
                scores['recall'] = self.calc_recall(preds, true)
            elif metric == 'f1':
                scores['f1'] = self.calc_f1(preds, true)
            else:
                raise ValueError(f"Given metric of %s must be one of 'accuracy','precision','recall','f1'" % metric)
        return scores
              

class LDA():
    def __init__(self):
        #LDA objects
        self.vector = None
        self.mu_1 = None
        self.mu_2 = None
        self.S = None
        self.B = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
    
        
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
        print('B calculated')
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
        print('S calculated')
        return
 

    def fit(self,data,y):
        self.get_stats(data,y)
        self.get_B(data,y)
        self.get_S(data,y)
        eig_values, eig_vectors = np.linalg.eig(np.matmul(np.linalg.inv(self.S), self.B))
        print('eigs calculated')
        self.vector = np.real(eig_vectors[eig_values.argmax()])


    def predict(self,x):
        preds = []
        if self.vector is None:
            print(f'Unable to make predictions until the model is fit. Please use fit() first.') 
            return
        else:
            for x_i in x:
                proj = np.dot(self.vector,x_i)
                if abs(proj - np.dot(self.vector, self.mu_1)) < abs(proj - np.dot(self.vector, self.mu_2)):
                    preds.append(1)
                else:
                    preds.append(-1)
        return preds
    
    def calc_accuracy(self, preds, true):
        self.accuracy = sum(preds == true) / true.shape[0]
        return self.accuracy
    
    def calc_precision(self, preds, true):
        true_positives = ((preds == 1) & (true == 1)).sum()
        self.precision = true_positives / (preds == 1).sum()
        return self.precision
    
    def calc_recall(self, preds, true):
        true_positives = ((preds == 1) & (true == 1)).sum()
        self.recall = true_positives / (true == 1).sum()
        return self.precision
    
    def calc_f1(self, preds, true):
        if self.precision == None:
            self.precision = self.calc_precision(preds, true)
        if self.recall == None:
            self.recall = self.calc_recall(preds, true)
        self.f1 = 2*(self.precision*self.recall) / (self.precision + self.recall)
        return self.f1
        
    
    def score(self, preds, true, metrics=['accuracy','precision','recall','f1']):
        scores = {}
        for metric in metrics:
            if metric == 'accuracy':
                scores['accuracy'] = self.calc_accuracy(preds, true)
            elif metric == 'precision':
                scores['precision'] = self.calc_precision(preds, true)
            elif metric == 'recall':
                scores['recall'] = self.calc_recall(preds, true)
            elif metric == 'f1':
                scores['f1'] = self.calc_f1(preds, true)
            else:
                raise ValueError(f"Given metric of %s must be one of 'accuracy','precision','recall','f1'" % metric)
        return scores

class SVM():
    def __init__(self):
        #SVM objects
        self.rate = None
        self.w = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
    
    def fit(self,train,labels, c=1, epoch=30, rate=.01):
        """
           This method actually builds the model.

           :parameter train - Pandas DataFrame containing explanatory variables
           :parameter labels - Pandas DataFrame containing response variables
           :parameter c - Hyparameter to influence support vectors
            Default is 1.
           :parameter epoch - Number of iterations for stochastic gradient descent
            Default is 10000
           :parameter rate - learning rate for gradient descent
            Default is .01
           """
        self.rate = rate
        self.w = np.zeros(train.shape[1])
        lam = -2*c/epoch
        for i in range(epoch):
            w_o = self.w
            for i, x in enumerate(train):
                val = x.dot(self.w)
                if (labels[i]*val < 1):
                    self.w = self.w + rate*((labels[i]*x) + lam*self.w)
                else:
                    self.w = self.w + rate*(lam*self.w)
        return
        
    def predict(self,data):
        """
        Makes predictions based on fit data. This means that fit() must be called before predict()

        :parameter data - Pandas DataFrame of data you want to make predictions about.
        """
        if self.w is None:
            print(f'Unable to make predictions until the model is fit. Please use fit() first.') 
            return
        predictions = []
        for x in data:
            projection = np.dot(x, self.w)
            predictions.append(projection)
        return np.sign(predictions)
    
    def calc_accuracy(self, preds, true):
        self.accuracy = sum(preds == true) / true.shape[0]
        return self.accuracy
    
    def calc_precision(self, preds, true):
        true_positives = ((preds == 1) & (true == 1)).sum()
        self.precision = true_positives / (preds == 1).sum()
        return self.precision
    
    def calc_recall(self, preds, true):
        true_positives = ((preds == 1) & (true == 1)).sum()
        self.recall = true_positives / (true == 1).sum()
        return self.recall
    
    def calc_f1(self, preds, true):
        if self.precision == None:
            self.precision = self.calc_precision(preds, true)
        if self.recall == None:
            self.recall = self.calc_recall(preds, true)
        self.f1 = 2*(self.precision*self.recall) / (self.precision + self.recall)
        return self.f1
        
    
    def score(self, preds, true, metrics=['accuracy','precision','recall','f1']):
        scores = {}
        for metric in metrics:
            if metric == 'accuracy':
                scores['accuracy'] = self.calc_accuracy(preds, true)
            elif metric == 'precision':
                scores['precision'] = self.calc_precision(preds, true)
            elif metric == 'recall':
                scores['recall'] = self.calc_recall(preds, true)
            elif metric == 'f1':
                scores['f1'] = self.calc_f1(preds, true)
            else:
                raise ValueError(f"Given metric of %s must be one of 'accuracy','precision','recall','f1'" % metric)
        return scores