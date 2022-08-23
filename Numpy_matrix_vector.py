# Kiem tra cai dat numpy
import numpy as np
print(np.pi)

#
# Matrix and vector
#Matrix = 2D array = list in Python
_A =[
		[1, 2, 3],
		[4, 5, 6],
		[7, 8 ,9]
   ]

print(_A)

#Vector = 1D array = list
_B = [1 , 2, 3]
print(_B)

#Khoi tao matrix in numpy: lay tu List OK
import numpy as np   
#np.array(object, dtype=None, ndmin=0)
A = np.array(_A)
print(A)
B = np.array(_B)
print(B)

#Indexing matrix, vector
print('a[0,0] = ', A[0,0])
print('a[0,1] = ', A[0,1])
print('a[1,:] = ', A[1,:])
print('a[:,2] = ', A[:,2])


#-------------------------------------
#Toan tu trong matrices
_A =[
		[1, 2, 3],
		[4, 5, 6],
		[7, 8 ,9]
   ]


_B =[
		[2, 4, 6],
		[3, 5, 7],
		[4, 6 ,8]
   ]

#Khoi tao matrix in numpy: lay tu List OK
import numpy as np   
#np.array(object, dtype=None, ndmin=0)
A = np.array(_A)
#print(A)
B = np.array(_B)
#print(B)

# +, -
print('A + B: \n', A + B)
print('A - B: \n', A - B)

# x*number
print('A * 2: \n', A*2)

# matrix*vector: m*n x n*1 = m*1
_C = [
		[1], 
		[2] , 
		[3]
	]
C = np.array(_C)
print('A*B: \n', A.dot(C))

#matrix*matrix: m*n x n*p = m*p
_D = [
		[1, 2], 
		[2, 3] , 
		[3, 4]
	]
D = np.array(_D)
print('A*D: \n', A@D) # A@D = A.dot(D)

#Identity matrix: duong cheo =1
E = np.eye(5);
print(E)


#Nhan 2 matrix cung size = nhan tung phan tu voi nhau:
print ("A*B: \n", A*B)


#----------------
#Toan tu logic
import numpy as np
A = np.eye(5)
print(A==1)
print(A*2)

#----------------
#Matran nghich dao: inverse matrix
import numpy as np
_A =[
		[1, 2, 3],
		[4, 5, 6],
		[7, 8 ,9]
   ]
A = np.array(_A)

A_i = np.linalg.pinv(A)

print(A_i)
print(A@A_i)
#-------------------
#Matran chuyen T
import numpy as np
_A =[
		[1, 2, 3],
		[4, 5, 6],
		[7, 8 ,9]
   ]
A = np.array(_A)

A_T = np.transpose(A)

print(A)
print(A_T)
#-------------------