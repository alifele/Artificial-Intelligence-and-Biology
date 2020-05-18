import numpy as np


class Hebb():
    
    def __init__(self):
        pass
        
    def fit(self, x,y):
        
        w = np.zeros((x.shape[1],y.shape[1]))
        for i in range(x.shape[0]):
            w += x[i].reshape((-1,1)) @ y[i].reshape((1,-1))

        self.w = w
             
        
        
    def predict(self, x_test, activation = 'bipolar'):
        '''
        activation can be 
        -> bipolar  -1,0,1  for less0, 0, more0
        -> binary    0,1    for less0, more0
        -> th
        '''
        prediction = []
    
        for elem in x_test:
            data = elem.reshape((1,-1))
            y_pred = data @ self.w

            if activation == 'bipolar':
                result = np.zeros(y_pred.shape)[0]

                result[y_pred[0] > 0] =1
                result[y_pred[0] < 0] =-1

            prediction.append(result)
            
        prediction = np.array(prediction)
        
        return prediction
        
