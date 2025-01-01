# Copyright (c) 2025 Alexander Steen Sparre Bille

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from sympy import symbols, Matrix


def rotational_matrix(n):
    """
    Creates a rotational matrix of size n x n.
    
    Args:
        n (int): The size of the matrix.
        
    Returns:
        np.ndarray: The rotational matrix.
    """
    if n <= 0:
        raise ValueError("Size of the rotational matrix must be a positive integer.")
    
    # Initialize an n x n zero matrix
    R = np.zeros((n, n), dtype=int)
    
    # Set the last column to the first row
    R[0, -1] = 1
    
    # Fill the sub-diagonal with 1s
    for i in range(1, n):
        R[i, i - 1] = 1
    
    return R


# Initialize matrices
A = np.zeros((40, 40))  # Full matrix CRC
I = np.eye(31)          # Identity matrix

# Rotational matrix
# R = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0],
# ])

R = rotational_matrix(8)
print(R)

U = np.zeros((8, 32))
U[0, 0] = 1  # Matrix with a 1 at the first position and all zeros afterwards

# CRC32 Generator polynomial
G = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0])

# Assemble the full matrix A
A[0:31, 1:32] += I  # Identity matrix inserted into full matrix A
A[32:40, 32:40] += R  # Rotational matrix inserted into full matrix A
A[32:40, 0:32] += U  # Matrix U inserted into full matrix A
A[31, 0:32] += G  # Generator polynomial bit value inserted into full matrix A

# Symbolic variables
R_syms = symbols('R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29 R30 R31')
M_syms = symbols('M7 M6 M5 M4 M3 M2 M1 M0')

M = Matrix([[*R_syms, *M_syms]])  # Concatenate R and M variables into a 1, 40 matrix


print(M.shape)

# Matrix A as a symbolic matrix
A_sym = Matrix(A)

# Compute A^8
A2 = A_sym**8

# Compute B = M * A^8
B = M * A2

# Print the result (symbolic expression for B)
print(B)


