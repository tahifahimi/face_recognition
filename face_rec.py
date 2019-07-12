from PIL import Image
import glob
import numpy
import scipy
from scipy.sparse.linalg import eigs
import matplotlib.pyplot  as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

N = 165

# read images turn to gray and calculating average
fBar = numpy.zeros((243*320,1),numpy.int)
S = numpy.zeros((243*320 , N),numpy.int)

counter = 0
for filename in glob.glob('D:/Algebra/Project/Image_Processing/training/subject*.gif'):
    im = Image.open(filename)
    gray_image = im.convert('L')
    resize_image = gray_image.resize((243, 320))
    imarr = numpy.array(im,numpy.int)
    imarr = numpy.reshape(imarr,(imarr.size,1))
    # print(S[:,[counter]].shape)
    S[:,[counter]]= imarr
    counter = counter+1
    fBar = fBar + imarr

fBar = fBar / N

# subtracting average from each matrix
# stranspose = numpy.transpose(S)
# fbar_transpose = numpy.transpose(fBar)
# print("fbar tra ")
# print(fbar_transpose.shape)
# print(stranspose.shape)
# print(stranspose[0].shape)
# print("%%%%")
# print(S[0].shape)
A = numpy.zeros((243*320 , N),numpy.int)
count =0
for i in range(N):
    A[:,[count]] = numpy.subtract(S[:,[i]], fBar)
    count = count+1

# A = numpy.transpose(A)
# print(numpy.transpose(A).shape)

# SVD for A
AT = numpy.transpose(A)
print(AT.shape)
ATA = numpy.dot(AT, A)
print(ATA.shape)

# eigenvalue, eigenvector = scipy.sparse.linalg.eigs(ATA)
eigenvalue, eigenvector = numpy.linalg.eig(ATA)
idx = numpy.argsort(eigenvalue)
eigenvalue = eigenvalue[idx]
eigenvector = eigenvector[:,idx]
print("####3")
print(eigenvalue.shape)
print(eigenvector.shape)

print("$$$$$$")
V = eigenvector
print(V.shape)
S = numpy.diag(eigenvalue)
print(S.shape)

U = numpy.zeros((243*320,N) , numpy.complex)
for o in range(N):
    av = numpy.dot(A,V[:,o])
    length = numpy.linalg.norm(av)
    av = av/length
    U[:,o] = av
print(U.shape)

# eigen images
U = U.astype(numpy.float64)
for p in range(10):
    plt.imshow(U[:,p].reshape(243,320) , cmap='gray')
    plt.show()
    # reshaped = numpy.reshape(U[:, p], (243, 320))
    # mm = Image.fromarray(reshaped)
    # mm.show()


# calculating Xr
rr =[]
Xr_space = []
for r in range(100,N):
    rr.append(r)
    ur = U[0:r,0:r]
    vr = V[0:r,0:r]
    sigma_r = S[0:r,0:r]
    x =numpy.dot(ur , numpy.dot(sigma_r,vr))
    ar = numpy.zeros((243*320,N),numpy.complex)
    ar[0:r,0:r] = x
    Xr_space.append(x)
    print("the shape of x")
    print(x.shape)

# creating feature vectors
# for training data
uuu = U[:,0:100]
Ut = numpy.transpose(uuu)
r =100
feature_vec = numpy.zeros((r, N), numpy.complex)
for w in range(N):
    x = numpy.dot(Ut,A[:,w])
    print("oooooo")
    print(x.shape)
    feature_vec[:,w] = x


# last part
# face recognition for the k=15
r = 15
main_u = U[:,0:15]
main_s = S[0:15,0:15]
main_v = V[:,0:15]
print("sizes :::")
print(main_u.shape)
print(main_s.shape)
print(main_v.shape)

# reading all data in data test
S_test = numpy.zeros((243*320,15))
fBar_test = numpy.zeros((243*320,1))
counter =0
for filename in glob.glob('D:/Algebra/Project/Image_Processing/test/subject*.gif'):
    im = Image.open(filename)
    gray_image = im.convert('L')
    resize_image = gray_image.resize((243, 320))
    imarr = numpy.array(im, numpy.int)
    imarr = numpy.reshape(imarr, (imarr.size, 1))
    S_test[:, [counter]] = imarr
    counter = counter+1
    fBar = fBar + imarr

fBar = fBar / 15
A_test = numpy.zeros((243*320 , 15),numpy.int)
count =0
for i in range(15):
    A_test[:,[count]] = numpy.subtract(S_test[:,[i]], fBar_test)
    count = count+1


AT_test = numpy.transpose(A_test)
ATA_test = numpy.dot(AT_test, A_test)
eigenval, eigenvec = numpy.linalg.eig(ATA_test)
idx_ = numpy.argsort(eigenval)
eigenval = eigenval[idx_]
eigenvec = eigenvec[:,idx_]

v_test = eigenvec
s_test = numpy.diag(eigenval)
u_test = numpy.zeros((243*320,15) , numpy.complex)
for o in range(15):
    av_ = numpy.dot(A_test,v_test[:,o])
    length = numpy.linalg.norm(av_)
    av_ = av_/length
    u_test[:,o] = av_
# ut_test = numpy.transpose(u_test)
# X_test = numpy.dot(u_test,A_test)

print("in regression")
A = A.astype(numpy.float64)
U = U.astype(numpy.float64)
# S = S.astype(numpy.float64)
kk = numpy.array((1,15))
regr = linear_model.LinearRegression()
regr.fit(A[:,0:15],U[:,0:15])
predict = regr.predict(A_test[:,0:15])
u_test = u_test.astype(numpy.float64)
error = mean_squared_error(u_test,predict)
plt.plot(kk,error)
# plt.plot(A_test[:,0:15],predict,color='black')
# plt.scatter(A_test[:,0:15],u_test[:,0:15],color='blue')
# plt.xticks(())
# plt.yticks(())
plt.show()

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(U[:,0:r],predict ))
