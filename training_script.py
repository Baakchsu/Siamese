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

import numpy as np
from tqdm import tqdm

n_classes = 2
#import os


npy=np.load('axis_dataset//strictly_labelled//test_for_fullsiamese.npy')#loads the training data
np.random.shuffle(npy)
X1 = np.array([i[1]/255 for i in npy]).reshape(-1,40000)
X2 = np.array([i[0]/255 for i in npy]).reshape(-1,40000) #Extracts features from the numpy dump and reshapes into (-1,784) flattened pixel array
Y = [i[2] for i in npy] 

Y=np.array(Y).reshape(-1,2)  

left_input = tf.placeholder('float32',[None,40000],name='left_input')
right_input= tf.placeholder('float32',[None,40000],name='right_input')
left_output = tf.placeholder('float32',[None,n_classes],name='left_embed')
right_output = tf.placeholder('float32',[None,n_classes],name='right_embed')
y = tf.placeholder('float32',name='Y')
is_training = tf.placeholder('bool',name='is_training')

keep_rate = 0.95
keep_prob = tf.placeholder(tf.float32,name="prob") #Placeholder for the dropout rate in the dropout layer
weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,64])/10,dtype='float32'),
               'W_conv2':tf.Variable(tf.random_normal([3,3,64,256])/10,dtype='float32'),
               
               'W_fc':tf.Variable(tf.random_normal([7*7*256,256])/10,dtype='float32'),
               'hid1':tf.Variable(tf.random_normal([256, 64])/10,dtype='float32'),
               
               'out':tf.Variable(tf.random_normal([64, n_classes]),dtype='float32')}

biases = {'b_conv1':tf.Variable(tf.random_normal([64])/10,dtype='float32'),
               'b_conv2':tf.Variable(tf.random_normal([256])/10,dtype='float32'),
               
               'b_fc':tf.Variable(tf.random_normal([256])/10,dtype='float32'),
               'hid1':tf.Variable(tf.random_normal([64])/10,dtype='float32'),
               
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
        #conv1 = tf.nn.dropout(conv1,keep_prob=dropout)
        conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        
        fc = tf.reshape(conv1,[-1, 7*7*256])       
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
        
        fc = tf.nn.relu(tf.matmul(fc, weights['hid1'])+biases['hid1'],name='embedding')
    return fc

left_output = conv_net(left_input,reuse=False,is_training=is_training,dropout=keep_prob)
right_output = conv_net(right_input,reuse=True,is_training=is_training,dropout=keep_prob)

l1_norm=tf.abs(tf.subtract(left_output,right_output))

fc = tf.matmul(l1_norm, weights['out'])
fc = tf.add(fc,biases['out'],name='output')


l2_regularizer = tf.contrib.layers.l2_regularizer(
  scale=0.003, scope=None)
weights = tf.trainable_variables()
regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)#L2 regularisation





cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc,labels=y) )
cost =  tf.add(regularization_penalty,cost)


optimizer = tf.train.AdamOptimizer(learning_rate=.002,name='train_op').minimize(cost) #optimiser to minimise the cost
# =============================================================================
# gvs = optimizer.compute_gradients(cost)
# capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
# optimizer = optimizer.apply_gradients(capped_gvs)    
# =============================================================================
hm_epochs = 1 #number of epochs
saver=tf.train.Saver() #defines the saver object
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
correct = tf.equal(tf.argmax(fc, 1), tf.argmax(y, 1)) #
accuracy = tf.reduce_mean(tf.cast(correct, 'float32'),name='accuracy')
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in tqdm(range(hm_epochs)):
        epoch_loss = 0
     
        batch=10
        
              
        upper=(len(Y)//10)-50
        
        print(len(X1),len(Y))
        for i in range(0,upper):
                #training is donein batches of ten 
                __,c=sess.run([optimizer, cost], feed_dict={left_input:X1[i*batch:(i+1)*batch,:],right_input:X2[i*batch:(i+1)*batch,:],y:Y[i*batch:(i+1)*batch,:],keep_prob:keep_rate,is_training:True})
                epoch_loss += c
                if i%50==0:
                    
                    print("Cost for this batch is: ",c)
                    sums =0
                    count = 0
                    for i in range(15):
                        out=sess.run(accuracy,feed_dict={left_input:X1[i:i+100,:],right_input:X2[i:i+100,:],y:Y[i:i+100],is_training:False,keep_prob:1})
                        sums += out
                        count += 1
                    avg_accuracy = sums/count
                    print('training accuracy:',avg_accuracy )
                    
        
        sums =0
        count = 0
        for i in range(5,0,-1):
            out=sess.run(accuracy,feed_dict={left_input:X1[i*-100:(i-1)*100,:],right_input:X2[i*-100:(i-1)*100,:],y:Y[i*-100:(i-1)*100,:],is_training:False,keep_prob:1})
            sums += out
            count += 1
        sums = sums/count
        print('test accuracy:',sums)#accuracy.eval({left_input:X3[-150:,:],right_input:X4[-150:,:],y:Y1[-150:,:],keep_prob:1,is_training:False}))
        
     #acc=accuracy.eval({left_input:X3[-150:,:],right_input:X4[-150:,:],y:Y1[-150:,:],keep_prob:1,is_training:False})   
        print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
        if sums>.95:
          saver.save(sess,"axis_dataset/model_params1/model") #saves the model       

   
    

    #saver.save(sess,"G:/axis_bank_challenge/sample_Signature/axis_dataset/model_params/model") #saves the model
    