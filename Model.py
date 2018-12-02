# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:26:57 2018

@author: Baakchsu
"""

import tensorflow as tf








import cv2
import numpy as np
input_image_size=200
import os
from tkinter import messagebox
import tkinter as tk
from tkinter import filedialog
input_image_size=200

#pickle_in.close()
print("****************Loading the Siamese model****************\n")

tf.reset_default_graph()
sess = tf.Session()
saver = tf.train.import_meta_graph('./model_params/model.meta')
saver.restore(sess,'./model_params/model')#/model')
graph = tf.get_default_graph()
left_input = graph.get_tensor_by_name("left_input:0")
right_input = graph.get_tensor_by_name("right_input:0")
y = graph.get_tensor_by_name('Y:0')
output = graph.get_tensor_by_name("output:0")
accuracy=graph.get_operation_by_name("accuracy").outputs[0]
is_training = graph.get_tensor_by_name("is_training:0")
prob = graph.get_tensor_by_name("prob:0")
print("\t\t Succesfully Loaded The Model \t\t")

root = tk.Tk()
 

messagebox.showinfo("Information","The program will prompt you to select two signature images that are to be compared and will display if they match or not.")

root.destroy()









root = tk.Tk()
answer1 = filedialog.askopenfilename(parent=root,
                                    initialdir=os.getcwd(),
                                    title="Please select a genuine signature image file")
answer2 = filedialog.askopenfilename(parent=root,
                                    initialdir=os.getcwd(),
                                    title="Please select a signature image that is to be tested")
root.destroy()                                                                               
try:
           
    sig1_img = cv2.imread(answer1,0)
    resized1 = np.array(cv2.resize(sig1_img, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)).reshape(-1,40000)
    resized1 = resized1/255
    sig2_img = cv2.imread(answer2,0)
    resized2 = np.array(cv2.resize(sig2_img, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)).reshape(-1,40000)
    resized2 = resized2/255
    out = sess.run(output,feed_dict={left_input:resized1,right_input:resized2,is_training:False,prob:.1})
    out=sess.run(tf.nn.softmax(out))
    label=''
    out = np.argmax(out)
    if out==1:
        label=False
    else:
                   label=True
    if label==True:
               root = tk.Tk()
               messagebox.showinfo("Result","The signatures match!")
               root.destroy()
    else:
               root = tk.Tk()
               messagebox.showwarning("Result","The two signatures do not match!")
               root.destroy()
           
except:
    print("Something went wrong! Please try again.")

    
    
    




      


   


# =============================================================================
# sums =0
# count = 0
# for i in range(21):
#     out=sess.run(accuracy,feed_dict={left_input:X1[i:i+100,:],right_input:X2[i:i+100,:],y:Y[i:i+100],is_training:False,prob:0})
#     sums += out
#     print(out)
#     count += 1
# sums = sums/count
# print('\n',sums)
# =============================================================================


