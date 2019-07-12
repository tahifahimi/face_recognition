# this is phase  1 of the project of linear algebra
# written by tahere fahimi
# version 4
# date : 4/11/1397
# the explanation of each part has been included by a pdf file

import cv2
import matplotlib.pyplot as plt
import numpy
from PIL import Image
import glob


N = 150

# read images turn to gray and calculating average
avg = numpy.zeros((243, 320), numpy.int)
image_list = []
for filename in glob.glob('D:/Algebra/Project/Image_Processing/training/subject*.gif'):
    im = Image.open(filename)
    gray_image = im.convert('L')
    resize_image = gray_image.resize((243, 320))
    imarr = numpy.array(im, numpy.int)
    image_list.append(imarr)
    # print(imarr.size)
    avg = avg + imarr
avg = avg / N
out_avg = Image.fromarray(avg)
out_avg = out_avg.convert('L')
out_avg.save('avg.png')
# out.show()

# subtracting average from each matrix
subtracted = []
for i in range(len(image_list)):
    subtracted.append(numpy.subtract(image_list[i], avg))
print(len(subtracted))
subtracted_view = Image.fromarray(subtracted[0])
subtracted_view = subtracted_view.convert('L')
subtracted_view.save('subtracted.png')
# subtracted_view.show()

# finding transpose of a matrix
transpose = []
for j in range(len(subtracted)):
    transpose.append(numpy.transpose(subtracted[j]))
# transpose_view = Image.fromarray(transpose[0])
# transpose_view = transpose_view.convert('L')
# transpose_view.save('transposed.jpg')
# transpose_view.show()

# finding the covariant
mul_cov = []
sum_cov = numpy.zeros((243, 243), numpy.int)
for l in range(len(subtracted)):
    mul_cov.append(numpy.dot(subtracted[l], transpose[l]))
    sum_cov = sum_cov + mul_cov[l]
covariant = sum_cov / N
covariant_view = Image.fromarray(covariant)
covariant_view = covariant_view.convert('L')
covariant_view.save('covariant.png')
# covariant_view.show()


# m1 = cv2.imread('avg.png')
# m2 = cv2.imread('subtracted.png')
# m4 = cv2.imread('covariant.png')
#
# plt.subplot(1,3,1)
# plt.imshow(m1)
#
# plt.subplot(1,3,2)
# plt.imshow(m2)
#
# plt.subplot(1,3,3)
# plt.imshow(m4)
# plt.show()

# finding eigenvalues and eigenvector
e_val ,e_vec = numpy.linalg.eig(covariant)
idx = numpy.argsort(e_val)
e_val = e_val[idx]
e_vec = e_vec[:,idx]
print(e_vec.shape)
print(e_val.shape)

#creating face data base
# face_rec = []
# v1 = numpy.reshape((243,1))

f = numpy.zeros((6,243, 320), numpy.float)
for z in range(6):
    v1 = e_vec[:, z]
    for q in range(N):          # sum all of multiple of v and Q
        k = numpy.full((243,320),v1[q])
        f[z] = f[z] + numpy.multiply(k,subtracted[q])

to_image = Image.fromarray(f[0])
to_image = to_image.convert('L')
to_image.save('1.png')

to_image = Image.fromarray(f[1])
to_image = to_image.convert('L')
to_image.save('2.png')

to_image = Image.fromarray(f[2])
to_image = to_image.convert('L')
to_image.save('3.png')

to_image = Image.fromarray(f[3])
to_image = to_image.convert('L')
to_image.save('4.png')

to_image = Image.fromarray(f[4])
to_image = to_image.convert('L')
to_image.save('5.png')

to_image = Image.fromarray(f[5])
to_image = to_image.convert('L')
to_image.save('6.png')

t1 = cv2.imread('1.png')
t2 = cv2.imread('2.png')
t3 = cv2.imread('3.png')
t4 = cv2.imread('4.png')
t5 = cv2.imread('5.png')
t6 = cv2.imread('6.png')

plt.subplot(2,3,1)
plt.imshow(t1)
plt.subplot(2,3,2)
plt.imshow(t2)
plt.subplot(2,3,3)
plt.imshow(t3)
plt.subplot(2,3,4)
plt.imshow(t4)
plt.subplot(2,3,5)
plt.imshow(t5)
plt.subplot(2,3,6)
plt.imshow(t6)

plt.show()
