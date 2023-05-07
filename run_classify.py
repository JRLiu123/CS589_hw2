#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from run import *
pos_train,neg_train,pos_test,neg_test, vocab = naive_bayes()


# In[ ]:


from create_matirx import *
from Multi_NB import *

# create the whole train_set

train = pos_train + neg_train
y0 = np.zeros((len(pos_train),1))
y1 = np.ones((len(neg_train),1))
train_y = np.vstack((y0,y1))

# convert the dataset into matrix

train_matrix = create_matrix(train)
collect_words = train_matrix.collectWords()
words_index = train_matrix.index_words(collect_words)
train_X = train_matrix.make_matrix(collect_words,words_index)


#create test_X
test_y_neg = np.ones((1,1)) 
neg_new_table = np.zeros((len(neg_test),train_X.shape[1]))
new_words_neg = np.zeros(len(neg_test))
b = 0

test_y_pos = np.zeros((1,1)) 
pos_new_table = np.zeros((len(pos_test),train_X.shape[1]))
new_words_pos = np.zeros(len(pos_test))
a = 0
 

for i in range(len(pos_test)):
    print('##########')
    print(i)
    pos_new_table[i],new_words_pos[i] = train_matrix.ins_table(pos_test[i],words_index) 
    Multi_NaiveBayes_pos = Multi_NB(train_X, train_y, pos_new_table[i], test_y_pos,new_words_pos[i],alpha = 1)
    accuracy_log_pos = Multi_NaiveBayes_pos.confusion_matrix()
    
    if accuracy_log_pos==1.0:
        a = a + 1
        print('a: ',a)
    
    print('------------------')
    neg_new_table[i],new_words_neg[i] = train_matrix.ins_table(neg_test[i],words_index)    
    Multi_NaiveBayes_neg = Multi_NB(train_X, train_y, neg_new_table[i], test_y_neg,new_words_neg[i],alpha = 1)
    accuracy_log_neg = Multi_NaiveBayes_neg.confusion_matrix()
    
    if accuracy_log_neg==1.0:
        b = b + 1
        print('b: ',b)  
    
print('the correct for pos is: ',a)
print('the correct for neg is :',b)    



# In[ ]:

'''
import matplotlib.pyplot as plt
import numpy as np


x = alpha
y1 = accuracy_log
y2 = accuracy


plt.plot(x, y1)
plt.scatter(x, y1)
#plt.plot(x, y2)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Model’s accuracy on the test set')
plt.ylim(0.4,1)

plt.grid(linestyle='-.')
for a,b in zip(x,y1):
#     plt.text(a,b-0.03,b,ha='center',va='bottom',fontsize=20)  
    plt.text(a,b+0.02,'%.3f' % b,ha='center',va='center',fontsize=10)  
'''

plt.tight_layout()
plt.show()

