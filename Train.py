# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:13:40 2018

@author: Baakchsu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:32:50 2018

@author: Baakchsu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 18:16:55 2018

@author: Baakchsu
"""









import tensorflow as tf
tf.reset_default_graph()
margin=1
import numpy as np
from tqdm import tqdm
import random
from tqdm import tqdm
n_classes = 2
#import os

np4=np.load('strictly_labelled\dataset4for_fullsiamese.npy')
X3 = np.array([i[1]/255 for i in np4]).reshape(-1,40000) 
X4 = np.array([i[0]/255 for i in np4]).reshape(-1,40000) #Extracts features from the numpy dump and reshapes into (-1,784) flattened pixel array
Y1 = [i[2] for i in np4]

Y1=np.array(Y1).reshape(-1,2)
#print(Y1[:5])

npy=np.load('G://axis_bank_challenge//sample_Signature//axis_dataset//strictly_labelled//test_for_fullsiamese.npy')
np.random.shuffle(npy)
X1 = np.array([i[1]/255 for i in npy]).reshape(-1,40000)
X2 = np.array([i[0]/255 for i in npy]).reshape(-1,40000) #Extracts features from the numpy dump and reshapes into (-1,784) flattened pixel array
Y = [i[2] for i in npy] 
upper=(len(X1)//20)-7
Y=np.array(Y).reshape(-1,2)  

left_input = tf.placeholder('float32',[None,40000],name='left_input')
right_input= tf.placeholder('float32',[None,40000],name='right_input')
left_output = tf.placeholder('float32',[None,n_classes],name='left_embed')
right_output = tf.placeholder('float32',[None,n_classes],name='right_embed')
y = tf.placeholder('float32',name='Y')
is_training = tf.placeholder('bool',name='is_training')

keep_rate = 0.9
keep_prob = tf.placeholder(tf.float32,name="prob") #Placeholder for the dropout rate in the dropout layer
weights = {'W_conv1':tf.Variable(tf.random_normal([7,7,1,64]),dtype='float32'),
               'W_conv2':tf.Variable(tf.random_normal([3,3,64,392]),dtype='float32'),
               'W_fc':tf.Variable(tf.random_normal([7*7*392,256]),dtype='float32'),
               'hid1':tf.Variable(tf.random_normal([256, 64]),dtype='float32'),
               
               'out':tf.Variable(tf.random_normal([64, n_classes]),dtype='float32')}

biases = {'b_conv1':tf.Variable(tf.random_normal([64]),dtype='float32'),
               'b_conv2':tf.Variable(tf.random_normal([392]),dtype='float32'),
               'b_fc':tf.Variable(tf.random_normal([256]),dtype='float32'),
               'hid1':tf.Variable(tf.random_normal([64]),dtype='float32'),
               
               'out':tf.Variable(tf.random_normal([n_classes]),dtype='float32')}



def conv_net(x,reuse, is_training,dropout):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        

        
        x1 = tf.reshape(x, shape=[-1, 200, 200, 1])
        
        conv1 = tf.nn.relu(tf.nn.conv2d(x1, weights['W_conv1'], strides=[1,1,1,1], padding='SAME') + biases['b_conv1'])
        conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        conv1 = tf.nn.relu(tf.nn.conv2d(conv1, weights['W_conv2'], strides=[1,1,1,1], padding='SAME') + biases['b_conv2'])
        
        conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        fc = tf.reshape(conv1,[-1, 7*7*392])       
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
        #fc = tf.nn.dropout(fc,keep_prob=dropout)
        fc = tf.nn.relu(tf.matmul(fc, weights['hid1'])+biases['hid1'],name='embedding')
    return fc

left_output = conv_net(left_input,reuse=False,is_training=is_training,dropout=keep_prob)
right_output = conv_net(right_input,reuse=True,is_training=is_training,dropout=keep_prob)
#chi_square#l2_norm=tf.divide(tf.square(left_output - right_output),tf.add(left_output,right_output))
l2_norm=tf.abs(tf.subtract(left_output,right_output))

fc = tf.matmul(l2_norm, weights['out'])
fc = tf.add(fc,biases['out'],name='output')


l2_regularizer = tf.contrib.layers.l2_regularizer(
  scale=0.001, scope=None)
weights = tf.trainable_variables()
regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)





cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc,labels=y) )
cost =  tf.add(regularization_penalty,cost)


optimizer = tf.train.AdamOptimizer(learning_rate=.005,name='train_op').minimize(cost) #optimiser to minimise the cost
# =============================================================================
# gvs = optimizer.compute_gradients(cost)
# capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
# optimizer = optimizer.apply_gradients(capped_gvs)    
# =============================================================================
hm_epochs = 3 #number of epochs
saver=tf.train.Saver() #defines the saver object
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
correct = tf.equal(tf.argmax(fc, 1), tf.argmax(y, 1)) #
accuracy = tf.reduce_mean(tf.cast(correct, 'float32'),name='accuracy')
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in tqdm(range(hm_epochs)):
        epoch_loss = 0
     
        batch=20
        
              
        
        #-(np.random.randint(low=1,high=30,size=(10,40000))/255)
        print(len(X1),len(Y))
        for i in range(0,upper):
                
                __,c=sess.run([optimizer, cost], feed_dict={left_input:X1[i*batch:(i+1)*batch,:],right_input:X2[i*batch:(i+1)*batch,:],y:Y[i*batch:(i+1)*batch,:],keep_prob:keep_rate,is_training:True})
                epoch_loss += c
                if i%20==0:
                    
                    print("Cost for this batch is: ",c)
                    sums =0
                    count = 0
                    for i in range(25):
                        out=sess.run(accuracy,feed_dict={left_input:X1[i:i+100,:],right_input:X2[i:i+100,:],y:Y[i:i+100],is_training:False,keep_prob:1})
                        sums += out
                        count += 1
                    avg_accuracy = sums/count
                    print('training accuracy:',avg_accuracy )
                    #print('Accuracy:',accuracy.eval({left_input:X1[-100:,:],right_input:X2[-100:,:],y:Y[-100:,:],keep_prob:1,is_training:False}))
        
        

        

    saver.save(sess,"G:/axis_bank_challenge/sample_Signature/axis_dataset/model_params/model") #saves the model
    