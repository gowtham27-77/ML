{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8a84580c-b749-459b-97ae-a062766eeb0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4612a836-d607-4d8b-9fe8-a8acf03f3be5",
   "metadata": {},
   "outputs": [],
   "source": [
    "#dir (np)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7659b990-f0de-4724-945a-9b29a5d702d3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['V' '40' '23' '5.5' '67']\n",
      "<class 'numpy.ndarray'>\n",
      "<U32\n"
     ]
    }
   ],
   "source": [
    "x = np.array([\"V\",40,23,5.5,67])\n",
    "print(x)\n",
    "print(type(x))\n",
    "print(x.dtype)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "090a730e-5d95-4213-b073-7959a7b94283",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[20 45]\n",
      " [78 98]]\n",
      "<class 'numpy.ndarray'>\n",
      "int32\n"
     ]
    }
   ],
   "source": [
    "#create a 2D array\n",
    "a1 = np.array([[20,45],[78,98]])\n",
    "print(a1)\n",
    "print(type(a1))\n",
    "print(a1.dtype)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "c050f01c-8b17-4bfc-960e-5f89f8fe2adf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 3.  4.  5.  6.]\n",
      " [ 7.  8.  9. nan]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "dtype('float64')"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Use astype() to convert the data type\n",
    "a1 = np.array([[3,4,5,6],[7,8,9,np.NAN]])\n",
    "print(a1)\n",
    "a1.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "639a010f-8a00-4362-889e-832c43d27892",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[['3.0' '4.0' '5.0' '6.0']\n",
      " ['7.0' '8.0' '9.0' 'nan']]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "dtype('<U32')"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a1_copy = a1.astype(str)\n",
    "print(a1_copy)\n",
    "a1_copy.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "c0ae16aa-97da-4144-95f7-4b0cc770eb47",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 3,  4,  5],\n",
       "       [ 7,  9, 10],\n",
       "       [ 4,  6, 12]])"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Mathematical operation on rows and columns\n",
    "a2 = np.array([[3,4,5],[7,9,10],[4,6,12]])\n",
    "a2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "48fe541b-cc19-42c5-bca1-d11e18a3765b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[12 26 22]\n",
      "[14 19 27]\n"
     ]
    }
   ],
   "source": [
    "print(a2.sum(axis = 1))\n",
    "print(a2.sum(axis = 0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "b0e68d3d-519c-456d-8d3d-8443a0f93b85",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 3  4  5]\n",
      " [ 7  9 10]\n",
      " [ 4  6 12]]\n",
      "[4.         8.66666667 7.33333333]\n",
      "[4.66666667 6.33333333 9.        ]\n"
     ]
    }
   ],
   "source": [
    "# find the mean values along row and colums\n",
    "print(a2)\n",
    "print(np.mean(a2, axis = 1))\n",
    "print(np.mean(a2, axis = 0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "cdb71f50-ba2f-4944-a06d-078f79bb93e0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 3  4  5]\n",
      " [ 7  9 10]\n",
      " [ 4  6 12]]\n",
      "[[ 0  4  5]\n",
      " [ 7  0 10]\n",
      " [ 4  6  0]]\n"
     ]
    }
   ],
   "source": [
    "#Matrix operations\n",
    "a3 = np.array([[3,4,5],[7,9,10],[4,6,12]])\n",
    "print(a3)\n",
    "np.fill_diagonal(a3,0)\n",
    "print(a3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "6e3ae213-e964-44b3-a7bf-6813e44b466a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# define two matices mutiple\n",
    "A = np.array([[20,45],[74,4]])\n",
    "B= np.array([[20,45],[78,98]])\n",
    "C = np.matmul(A,B)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "5942f36c-b70b-4072-957a-97932f8fec13",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[20 74]\n",
      " [45  4]]\n",
      "[[20 78]\n",
      " [45 98]]\n"
     ]
    }
   ],
   "source": [
    "#print the Transpose of the matrices\n",
    "print(A.T)\n",
    "print(B.T)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "26114dd8-1825-439a-bb5a-0877f6790d59",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 3,  4,  5],\n",
       "       [ 7,  8,  9],\n",
       "       [ 9,  1,  6],\n",
       "       [10,  9, 18]])"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#### Accessing the array elements\n",
    "a4 = np.array([[3,4,5],[7,8,9],[9,1,6],[10,9,18]])\n",
    "a4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "c54d7c03-7c42-47bc-b8d8-c3038cd292f9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[4],\n",
       "       [8],\n",
       "       [1],\n",
       "       [9]])"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#slicing of array\n",
    "a4[2:3:,1:2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4698bc90-5ca4-40f2-8977-7be27d664209",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
