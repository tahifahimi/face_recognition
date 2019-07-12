# phase 3
# SVD compression
# written by tahere fahimi
# date : 28/10/1397
# compress a random Image

import numpy
from PIL import Image
import matplotlib.pyplot as plt

# SVD compression by package
svd =[]
k_val = []
ax = numpy.array(Image.open('subject01.jpg'),numpy.float)
k=1
u, s, v = numpy.linalg.svd(ax, full_matrices=True)
while k<=106 :
    k_val.append(k)
    print(u.shape)
    print(numpy.diag(s[:k]).shape)
    print(v.shape)
    reconst_matrix = numpy.dot(u[:,:k],numpy.dot(numpy.diag(s[:k]),v[:k,:]))
    reconst_matrix = reconst_matrix.astype(numpy.uint8)
    svd.append(reconst_matrix.sum()/ax.sum())
    reconst_matrix = Image.fromarray(reconst_matrix)
    reconst_matrix.show()
    k =k +25

plt.plot(k_val,svd)
plt.show()



# the quality of compression is :
# = 10 log((max.range)2 / Root mean square error)

#svd compression by code
# finding sigma
# main_sigma is the difference of the sigma and sigmaK
# ax = numpy.array(Image.open('subject01.jpg'),numpy.int)
# axt = numpy.transpose(ax)
# AtA = numpy.dot(axt,ax)
#
#
# eigenValues, eigenVectors = numpy.linalg.eig(AtA)
#
# idx = eigenValues.argsort()[::-1]
# eigenValues = eigenValues[idx]
# eigenVectors = eigenVectors[:,idx]
#
#
# # # print(singular)
# # n= len(singular)
# # for i in range(n):
# #     for j in range(0, n - i - 1):
# #         if abs(singular[j]) > abs(singular[j + 1]):
# #             if (i==1) :
# #                 print(singular[j])
# #                 print(singular[j+1])
# #             singular[j], singular[j + 1] = singular[j + 1], singular[j]
# #             if (i == 1):
# #                 print(singular[j])
# #                 print(singular[j + 1])
# #             vec[:,[j,j+1]] = vec[:, [j+1,j]]
#
#
# r = math.floor(len(eigenValues))
# singular = numpy.diag(eigenValues)
#
# s_v = []
# for i in range(r):
#     s_v.append(singular[i])
#
# sigma = numpy.diag(numpy.diag(s_v))
# main_sigma = numpy.zeros((320,320) , dtype=numpy.complex)
# main_sigma[:sigma.shape[0], :sigma.shape[1]] = sigma
# # print(main_sigma)
# if main_sigma.all() == sigma.all() :
#     print("true")
#
# # finding Vtranspose
# vT = numpy.transpose(eigenVectors)
# print("v shapev:")
# print(vT.shape)
#
# # finding u
# temp = numpy.dot(ax , vT)
# u = numpy.dot(temp , numpy.transpose(main_sigma))
# print("the u shape  : ")
# print(u.shape)
#
#
# # creating Dk
# d = numpy.dot(u,main_sigma)
# Dk = numpy.dot(d,vT)
# print(type(Dk))
#
# transpose_view = Image.fromarray(Dk,'RGB')
# transpose_view.save('mmm.jpg')
# transpose_view.show()
# # print(Dk.shape)