import numpy as np
import math
import operator
import sklearn.linear_model
from sklearn.datasets import load_digits
from sklearn import datasets
from sklearn import neighbors
from sklearn import metrics
from sklearn.metrics import accuracy_score

class NeuralNetwork:

    def __init__(self, structure,alpha, epoches):
        self.epoches = epoches
        self.alpha = alpha
        self.structure = structure
        self.gradient_w = []
        self.gradient_b = []
        self.sz = len(structure)
        self.a = []
        self.z = []
        self.b = []
        self.w = []
        for x in self.structure:
            self.a.append(np.empty((x,1)))
            self.z.append(np.empty((x,1)))
            self.b.append(np.random.rand(x,1))
            self.gradient_b.append(np.zeros((x,1)))

        for i in range(0,self.sz - 1):
            self.w.append(np.random.rand(self.structure[i+1], self.structure[i]) - 0.5)
            self.gradient_w.append(np.zeros((self.structure[i+1], self.structure[i])))

    def sigmoid(self, Z):
        return 1.0/(1.0 + np.exp(-Z))

    def der_sigmoid(self, Z):
        return self.sigmoid(Z)*(1-self.sigmoid(Z))


    def feedforward(self, X):
        self.a[0] = np.array(X);
        self.a[0] = np.reshape(self.a[0],(self.structure[0],1))

        for i in range(1,self.sz):    
            self.z[i] =  self.w[i-1]@self.a[i-1] + self.b[i]
            self.a[i] = self.sigmoid(self.z[i])

    def backpropagation(self, Y):
          Y = np.reshape(Y,(Y.size,1))
          L = self.sz - 1
          err = (self.a[L] - Y) * self.der_sigmoid(self.z[L])

          self.gradient_w[L-1]+=err@np.transpose(self.a[L-1])
          self.gradient_b[L]+= err

          for l in range(L-1,0,-1):
               err = (np.transpose(self.w[l])@err)*self.der_sigmoid(self.z[l]) 
               self.gradient_w[l-1]+=(err@np.transpose(self.a[l-1]))
               self.gradient_b[l]+=err
 


    def train(self, X,Y):
        for i in range(0, self.epoches):
            err = 0
            for j in range(0,self.sz):
                self.gradient_b[j] = np.zeros((self.structure[j],1))

            for j in range(0,self.sz - 1):
                self.gradient_w[j] = np.zeros((self.structure[j+1], self.structure[j]))

            for j in range(0,len(X)):
                self.feedforward(X[j])
                y = np.reshape(Y[j],(Y[j].size,1))
                err+=(np.transpose((self.a[self.sz-1] - y))@(self.a[self.sz-1] - y))
                
                self.backpropagation(Y[j])

            for j in range(0,self.sz - 1):
                self.w[j]-= self.alpha*(self.gradient_w[j]/len(X))

            for j in range(0,self.sz):
                self.b[j]-=self.alpha*(self.gradient_b[j]/len(X))

            err/= 2*len(X)

            print(err)

    def predict(self, X):
        Y = []
        for x in X:
            self.feedforward(x)

            Y.append(np.argmax(self.a[self.sz - 1]))
        Y = np.array(Y)
        Y = np.reshape(Y,(len(Y),1))
        return Y
            
