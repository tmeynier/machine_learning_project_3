#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 

import scipy.ndimage
import matplotlib.pyplot as plt


# In[2]:


def load_data(filepath, start_index, end_index):

    data = []
    for i in range(start_index,end_index):
        filename = filepath + 'sample-' + str(i) + '.npy'
        image_3d = np.load(filename)

        if i == start_index:
            data = image_3d
        else:
            data = np.concatenate((data, image_3d), axis=0)
    
    return data


# In[3]:


nb_images_train = 50
x_train = load_data('../data/train_images/', 0, nb_images_train)
y_train = load_data('../data/train_labels/', 0 , nb_images_train)


# In[4]:


#def rotate_data(x_original , y_original):
#    x_train_new = []
#    y_train_new = []
#    for i in range(x_original.shape[0]) : 
#        rand_a = np.random.uniform(-180,180)
#        x_normal = x_original[i, : , :]
#        x_rotated = scipy.ndimage.rotate(x_normal , angle=rand_a , reshape = False, order=1)
#        y_normal = y_original[i,:,:]
#        y_rotated = scipy.ndimage.rotate(y_normal, angle=rand_a , reshape=False, order=1)
#        x_normal = x_normal[np.newaxis, :,:]
#        x_rotated = x_rotated[np.newaxis, :,:]
#        y_normal = y_normal[np.newaxis, :,:]
#        y_rotated = y_rotated[np.newaxis, :,:]
#        if(i == 0) : 
#            x_train_new = np.concatenate((x_normal, x_rotated), axis = 0)
#            y_train_new = np.concatenate((y_normal, y_rotated), axis=0)

        #else : 
#            x_train_new = np.concatenate((x_train_new, x_normal, x_rotated), axis=0)
#            y_train_new = np.concatenate((y_train_new, y_normal, y_rotated), axis=0)
#    return x_train_new , y_train_new

num_rot = 3
def rotate_data(x_original , y_original):
    x_train_new = []
    y_train_new = []
    for i in range(x_original.shape[0]) : 
        rand_a = np.random.uniform(-180,180)
        x_normal = x_original[i, : , :]
        y_normal = y_original[i,:,:]

        if(i == 0) : 
            x_train_new = np.array(x_normal[np.newaxis, :,:])
            y_train_new = np.array(y_normal[np.newaxis, :,:])
        else : 
            x_train_new = np.concatenate((x_train_new, x_normal[np.newaxis, :,:]), axis=0)
            y_train_new = np.concatenate((y_train_new, y_normal[np.newaxis, :,:]), axis=0)            
            
        for j in range(num_rot):   
            x_rotated = scipy.ndimage.rotate(x_normal , angle=rand_a , reshape = False , order=1 )
            y_rotated = scipy.ndimage.rotate(y_normal, angle=rand_a , reshape=False,order=1)
            #x_normal = x_normal[np.newaxis, :,:]
            x_rotated = x_rotated[np.newaxis, :,:]
            #y_normal = y_normal[np.newaxis, :,:]
            y_rotated = y_rotated[np.newaxis, :,:]
            
            #print(x_train_new.shape)
            #print(x_rotated.shape)
            x_train_new = np.concatenate((x_train_new, x_rotated), axis=0)
            y_train_new = np.concatenate((y_train_new, y_rotated), axis=0)
    return x_train_new , y_train_new



# In[5]:


x_train_new, y_train_new = rotate_data(x_train,y_train)


# In[6]:


print(x_train_new.shape)
for i in range(10) :
    normal = x_train_new[i,:,:]
    rotated = y_train_new[i,:,:]
    fig = plt.figure(figsize=(8, 2))
    ax1, ax2= fig.subplots(1, 2)
    ax1.imshow(normal)
    ax2.imshow(rotated)


# In[ ]:


#Saving the results in numpy array :
np.save('../data/x_train_new.npy', x_train_new)
np.save('../data/y_train_new.npy', y_train_new)


# In[7]:


#Printing a test image to compare what we obtained

i = 51
x_t = np.load('../data/test_images/sample-'+str(i)+'.npy')
x_t_rot = np.load('../data/test_images_randomly_rotated/sample-'+str(i)+'.npy')



fig = plt.figure(figsize=(8, 2))
ax1, ax2= fig.subplots(1, 2)

ax1.imshow(x_t[0,:,:])
ax2.imshow(x_t_rot[0,:,:])


print(x_t[0,:,:].shape)
print(x_t_rot[0,:,:].shape)


# In[ ]:




