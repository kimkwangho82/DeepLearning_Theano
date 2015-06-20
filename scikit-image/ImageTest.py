
# In[23]:

from scipy import misc
l = misc.lena()
misc.imsave('lena.png', l) # uses the Image module (PIL)

import matplotlib.pyplot as plt
plt.imshow(l, cmap='gray')
plt.show()


# Out[23]:

# image file:

# In[21]:

lena = misc.imread('lena.png')


# In[3]:

lena


# Out[3]:

#     array([[159, 159, 159, ..., 168, 151, 119],
#            [159, 159, 159, ..., 168, 151, 119],
#            [159, 159, 159, ..., 168, 151, 119],
#            ..., 
#            [ 21,  21,  29, ...,  92,  87,  85],
#            [ 22,  22,  35, ...,  92,  93,  96],
#            [ 22,  22,  35, ...,  92,  93,  96]], dtype=uint8)

# In[4]:

test = misc.imread('Chrysanthemum.jpg')


# In[5]:

test


# Out[5]:

#     array([[[255,  67,   1],
#             [250,  65,   0],
#             [240,  69,   0],
#             ..., 
#             [251,  66,  20],
#             [239,  51,   6],
#             [239,  39,   5]],
#     
#            [[255,  66,   2],
#             [252,  64,   1],
#             [240,  69,   0],
#             ..., 
#             [252,  58,  20],
#             [236,  42,   7],
#             [241,  36,   5]],
#     
#            [[255,  68,   1],
#             [254,  70,   0],
#             [243,  66,   0],
#             ..., 
#             [245,  63,   0],
#             [239,  47,   0],
#             [234,  36,   0]],
#     
#            ..., 
#            [[212,  40,   0],
#             [212,  32,   0],
#             [210,  26,   2],
#             ..., 
#             [204,  19,   1],
#             [208,  19,   0],
#             [209,  25,   1]],
#     
#            [[220,  41,   0],
#             [216,  40,   1],
#             [220,  36,   2],
#             ..., 
#             [203,  17,   2],
#             [206,  20,   0],
#             [210,  24,   1]],
#     
#            [[222,  48,   0],
#             [223,  48,   1],
#             [227,  53,   0],
#             ..., 
#             [203,  17,   2],
#             [206,  19,   0],
#             [205,  25,   2]]], dtype=uint8)

# In[6]:

plt.imshow(test)
plt.show()


# Out[6]:

# image file:

# In[7]:

from skimage.color import rgb2gray


# In[8]:

test_gray = rgb2gray(test)


# In[24]:

plt.imshow(test_gray, cmap='gray')
plt.show()


# Out[24]:

# image file:

# In[10]:

misc.imsave('test_gray.jpg', test_gray)


# In[11]:

from skimage.transform import resize


# In[12]:

test_gray_resize = resize(test_gray, (32, 32))


# In[13]:

misc.imsave('test_gray_resize.jpg', test_gray_resize)


# In[14]:

temp = misc.imread('test_gray_resize.jpg')


# In[15]:

temp


# Out[15]:

#     array([[222,  89, 207, ...,  34, 150,  59],
#            [202, 140, 193, ..., 139,  20,  99],
#            [161, 220, 193, ..., 107,  52,  16],
#            ..., 
#            [ 40, 168, 172, ...,  21,  64,  85],
#            [162, 158, 180, ...,  48,  29,  95],
#            [ 43, 226, 150, ...,  58,  61,  44]], dtype=uint8)

# In[16]:

temp.flatten()


# Out[16]:

#     array([222,  89, 207, ...,  58,  61,  44], dtype=uint8)

# In[17]:

temp


# Out[17]:

#     array([[222,  89, 207, ...,  34, 150,  59],
#            [202, 140, 193, ..., 139,  20,  99],
#            [161, 220, 193, ..., 107,  52,  16],
#            ..., 
#            [ 40, 168, 172, ...,  21,  64,  85],
#            [162, 158, 180, ...,  48,  29,  95],
#            [ 43, 226, 150, ...,  58,  61,  44]], dtype=uint8)

# In[18]:

temp2 = temp.flatten()
temp2 = temp2/256.


# In[19]:

temp2


# Out[19]:

#     array([ 0.8671875 ,  0.34765625,  0.80859375, ...,  0.2265625 ,
#             0.23828125,  0.171875  ])

# In[ ]:



