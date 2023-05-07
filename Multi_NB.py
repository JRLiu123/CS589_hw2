#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math

#train model using Multi_NB method


#修改
import math
class Multi_NB():
    def __init__(self, train_X, train_y, test_X, test_y,new_words,alpha):
        
        self.train_X = train_X
        self.train_y = train_y
        self.test_X  = test_X 
        self.test_y  = test_y 
        self.new_words  = new_words
        self.alpha  = alpha
        
        

    def train(self):
        
        # select the reviews belonging to class0(pos)
     
        i_class0 = np.where(self.train_y == 0)[0]
        train_X_pos = self.train_X[i_class0]
        
        i_class1 = np.where(self.train_y == 1)[0]
        train_X_neg = self.train_X[i_class1]
        
        # count the number of each words happening in every class
        n_wk = np.zeros((2,self.train_X.shape[1]))
        n_wk[0] = train_X_pos.sum(axis=0)
        n_wk[1] = train_X_neg.sum(axis=0)
        
        # calculate the P of each words happening in every class
        P_wk = np.zeros((self.train_X.shape[1],2))
        
        P_wk[:,0] = (n_wk[0] + self.alpha)/(n_wk[0].sum() + self.alpha*self.train_X.shape[1])
        P_wk[:,1] = (n_wk[1] + self.alpha)/(n_wk[1].sum() + self.alpha*self.train_X.shape[1])
        #P_wk[:,0] = n_wk[0,:]/n_wk[0,:].sum()
        #P_wk[:,1] = n_wk[1,:]/n_wk[1,:].sum()
 
        #print('P_wk finished')
        # calculate the prior P of each class
        P_yi = np.zeros(2)
        P_yi[0] = len(train_X_pos)/len(self.train_X)
        P_yi[1] = len(train_X_neg)/len(self.train_X)

        
        return P_wk, P_yi,n_wk
      
      
    def predict(self):
        
   
        P_wk, P_yi,n_wk = self.train()
        P_media_1 = np.zeros((test_X.shape))
        P_media_2 = np.zeros((test_X.shape))
        P_test_X = np.ones((self.test_X.shape[0],2))
        
        
        #--------------------------------------------------------------------------
        laplace = np.zeros((self.test_X.shape[0],2))
        laplace[:,0] = self.alpha/(n_wk[0].sum() + self.alpha*self.train_X.shape[1])
        laplace[:,1] = self.alpha/(n_wk[1].sum() + self.alpha*self.train_X.shape[1])
        
        for k in range(test_X.shape[0]):
            laplace[k,0] = math.pow(laplace[k,0],self.new_words[k])
            laplace[k,1] = math.pow(laplace[k,1],self.new_words[k])
             
        #--------------------------------------------------------------------------
        
        
        
        for i in range(test_X.shape[0]):
            P_media_1[i] = np.multiply(self.test_X[i],P_wk[:,0])
            P_media_1[i][P_media_1[i]==0] = 1
            P_media_2[i] = np.multiply(self.test_X[i],P_wk[:,1])
            P_media_2[i][P_media_2[i]==0] = 1
            for j in range(test_X.shape[1]):
                P_test_X[i,0] = P_test_X[i,0]*P_media_1[i][j]
                P_test_X[i,1] = P_test_X[i,1]*P_media_2[i][j]
                
        
        P_test_X[:,0] = P_test_X[:,0]*P_yi[0]*laplace[:,0]
        P_test_X[:,1] = P_test_X[:,1]*P_yi[1]*laplace[:,1]
        
        print('P_test_X finished')

        #select the most possible label of test_X
        result = np.argmax(P_test_X, axis=1)
        
        return result
    
    
    def train_log(self):
        
        # select the reviews belonging to class0(pos)
        i_class0 = np.where(self.train_y == 0)[0]
        train_X_pos = self.train_X[i_class0]
        
        i_class1 = np.where(self.train_y == 1)[0]
        train_X_neg = self.train_X[i_class1]
        #print('1 finshed')
        # count the number of each words happening in every class
        n_wk = np.zeros((2,self.train_X.shape[1]))
        n_wk[0] = train_X_pos.sum(axis=0)
        n_wk[1] = train_X_neg.sum(axis=0)
        #print('2 finshed')
        # calculate the P of each words happening in every class
        P_wk = np.zeros((self.train_X.shape[1],2))
        
        P_wk[:,0] = (n_wk[0] + self.alpha)/(n_wk[0].sum() + self.alpha*self.train_X.shape[1])
        P_wk[:,1] = (n_wk[1] + self.alpha)/(n_wk[1].sum() + self.alpha*self.train_X.shape[1])
        #P_wk[:,0] = n_wk[0,:]/n_wk[0,:].sum()
        #P_wk[:,1] = n_wk[1,:]/n_wk[1,:].sum()
        P_wk_log = np.nan_to_num(np.log(P_wk), neginf=0) 
        #print('3 finshed')
        # calculate the prior P of each class
        P_yi = np.zeros(2)
        P_yi[0] = len(train_X_pos)/len(self.train_X)
        P_yi[1] = len(train_X_neg)/len(self.train_X)
        P_yi_log = np.log(P_yi)
        #print('4 finshed')
        # combine prior P with P_wk
        P_y_log = np.vstack((P_yi_log,P_wk_log))
        
        # combine vector with test_X:  with the aim of np.dot()
        add_col = np.ones(1)
        test_X_add = np.hstack((add_col, self.test_X))
        test_X_add = test_X_add.reshape(-1, 1).T
        #print('train_log finshed')

        
        return test_X_add, P_y_log,n_wk

          
    
    def predict_log(self):
        #print('predict starts')
        #test_X_add, P_y_log = self.train()
        test_X_add, P_y_log,n_wk = self.train_log()
        
        #calculate the P of pos and neg in test_X
        P_test_X = np.zeros((1,2))
        P_test_X = np.dot(test_X_add,P_y_log)
        #print('multiply finished')
        #---------------------------------------------------------------------------
        #handle the new words now showing in train_X
        laplace = np.zeros((1,2))
        laplace[0,0] = self.alpha/(n_wk[0].sum() + self.alpha*self.train_X.shape[1])
        laplace[0,1] = self.alpha/(n_wk[1].sum() + self.alpha*self.train_X.shape[1])
        
        P_test_X[0,0] = P_test_X[0,0] + self.new_words*np.log(laplace[0,0])
        P_test_X[0,1] = P_test_X[0,1] + self.new_words*np.log(laplace[0,1])
        #---------------------------------------------------------------------------------
        #select the most possible label of test_X
        result = np.argmax(P_test_X, axis=1)
        #print('predict finished')
        return result
        

    
    def confusion_matrix(self,method='log'):
        
        if method=='log':
        
            result = self.predict_log()
        else:
            result = self.predict()
        
        #convert the shape of test_y from n*1 to 1*n
        test_y = self.test_y.reshape(1, -1)
        
        #compare the predict and real results 
        accuracy = 0
        compare = result - test_y
        if compare[0]== 0:
            accuracy = 1.0
   
        '''
        c1 = len(np.where(test_y[0]==0)[0])
        c2 = len(np.where(test_y[0]==1)[0])
        
        confusion_matrix = np.zeros((2,2))
        confusion_matrix[0,1] = len(np.where(compare[0]== 1)[0])
        confusion_matrix[1,0] = len(np.where(compare[0]== -1)[0])
        confusion_matrix[0,0] = c1 - confusion_matrix[0,1]
        confusion_matrix[1,1] = c2 - confusion_matrix[1,0]
        
        sum_counts = confusion_matrix.sum()
        accurate = (confusion_matrix[0,0]+confusion_matrix[1,1])/sum_counts
        
        precision = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
        
        recall = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
        
        print('confusion matrix finished')
        return confusion_matrix,accurate,precision,recall
        '''
        return accuracy

