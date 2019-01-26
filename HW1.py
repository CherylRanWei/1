from __future__ import division
import cv2
import numpy as np
from skimage import img_as_ubyte


'''
For each question, uncomment it and comment other questions' code
Because image display with cv2 cannot stop
'''


####print '********** Problem 2 ************'
####'''
#### (a)
####'''
####img = cv2.imread('1_4.bmp')
####cv2.imshow('image', img)
####
####'''
#### (b)
####'''
####print 'Type of Image 1_4.bmp is:',img.dtype
####
####max_value = np.amax(img)
####print 'Maximum Data value for Image 1_4.bmp is:', max_value
####
####min_value = np.amin(img)
####print 'Minimum Data value for Image 1_4.bmp is:', min_value
####
####'''
#### (c)
####'''
####new_img = img.astype(np.float) /255
####print 'Image type after change:', new_img.dtype
####out = img_as_ubyte(new_img)
####print 'Image type after convert back:', out.dtype
####
####cv2.imshow('Convert Back Image', out)
####cv2.imwrite('Convert Back Image.png', out)
####cv2.waitKey(0)
####cv2.destroyAllWindows()
####
####print '********** Problem 3 ************'
####'''
#### (a)
####'''
####X = cv2.imread('1_2.tif')
####Y = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
####height, width = Y.shape
####
####cv2.imshow('gray_X', Y)
####cv2.imwrite('Gray X.png', Y)
####
####
####'''
#### (b)
####'''
####def imRotate(X, degrees):
####    '''Args:
####        X: numpy array of shape [height, width, channels]
####        degrees: rotation angle, positive means counter clockwise rotation Output: rotated image
####    '''
####    height, width= X.shape
####    M = cv2.getRotationMatrix2D((height/2.0,width/2.0), degrees, 1)
####    return cv2.warpAffine(X, M, (height, width))
####
####
####Z0 = imRotate(Y, 120)
####cv2.imshow('Z0', Z0)
####cv2.imwrite('Z0.png', Z0)
####
####
####'''
#### (c)
####'''
####Z1 = imRotate(Y, 10)
####
####i = 1
####while(i<12):
####    Z1 = imRotate(Z1, 10)
####    i += 1
####    
####cv2.imshow('Z1', Z1)
####cv2.imwrite('Z1.png', Z1)
####cv2.waitKey(0)
####cv2.destroyAllWindows()


print '**********Problem 4a ************' 
print '*** i ***'
X = np.loadtxt('1_3.asc') /255
print'Size of X is:',X.shape

Y1 = X[0:384:4,0:256:4]
print'Size of Y1 is:',Y1.shape

print  '*** ii ***' 
neighbor_size = 4
Y2_prime = np.zeros(X.shape)
Y2 = np.zeros(Y1.shape)
copy_img = np.zeros(X.shape)

def avg(img, x, y, size):
    ttl = 0
    for i in range(size):
        for j in range(size):
            #print(img[x+i, y+j])
            ttl = ttl + img[x+i, y+j]
    result = ttl / (size*size)
    return result


for x in range(0,X.shape[0],4):
    for y in range(0,X.shape[1],4):
        #print('****')
        copy_img[x,y] = avg(X, x, y, neighbor_size)
Y2 = copy_img[0:384:4,0:256:4]

print'Size of Y2 is:',Y2.shape

cv2.imshow('X', X)
cv2.imshow('Y1', Y1)
cv2.imshow('Y2', Y2)


print '**********Problem 4b ************'
print '*** i ***'
resized_Y1 = np.zeros(X.shape)
print Y1.shape
print resized_Y1.shape
neighbor_size = 4

def repeating(x, y, size):
    for i in range(size):
        for j in range(size):
            resized_Y1[x+i, y+j] = Y1[x//size,y//size]
            
for x in range(0,resized_Y1.shape[0], neighbor_size):   #0-383  rows
    for y in range(0,resized_Y1.shape[1], neighbor_size):  #0-255 cols
        repeating(x, y, neighbor_size)

print '*** ii ***'

neighbor_size = 4
empty_Y1 = np.zeros(X.shape)
empty_Y1[0:384:4,0:256:4] = Y1

for x in range(empty_Y1.shape[0]):   #0-383  rows
    for y in range(empty_Y1.shape[1]):  #0-255 cols
        x0 = x - (x % neighbor_size)
        x1 = x + (neighbor_size - (x % neighbor_size))
        y0 = y - (y % neighbor_size)
        y1 = y + (neighbor_size - (y % neighbor_size))

        if(x0 > empty_Y1.shape[0]-1  or y0 > empty_Y1.shape[1]-1  or x1 > empty_Y1.shape[0]-1 or y1 > empty_Y1.shape[1]-1):
            empty_Y1[x,y] = 0.0
        else:  
            f_R1 = empty_Y1[x0,y0] * ((y1-y)/(y1-y0)) + empty_Y1[x0,y1] * ((y-y0)/(y1-y0))
            f_R2 = empty_Y1[x1,y0] * ((y1-y)/(y1-y0)) + empty_Y1[x1,y1] * ((y-y0)/(y1-y0))
            empty_Y1[x,y] = f_R1 * ((x1-x)/(x1-x0)) + f_R2 * ((x-x0)/(x1-x0))

print 'Size of enlarged Y1 by pixel repeating is:',resized_Y1.shape
print 'Size of enlarged Y1 by bilinear interpolation is:',empty_Y1.shape
cv2.imshow('Pixel Repeating Y1', resized_Y1)
cv2.imshow('Bilinear Interpolation Y1', empty_Y1)
cv2.waitKey(0)
cv2.destroyAllWindows()


























