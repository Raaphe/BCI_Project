
---


> Most of these notes are provided by Microsoft [here](https://quantum.microsoft.com/en-us/experience/quantum-katas) 
> Another good source that details much found on here can be found on [this page](https://en.wikipedia.org/wiki/Determinant)

A **matrix** is set of numbers arranged in a rectangular grid. Here is a 2 by 2 matrix:
$$A =
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$
The notation $A_{i,j}$ refers to the element in row 𝑖 and column 𝑗 of matrix 𝐴 (all indices are 0-based). In the above example, $A_{0,1} = 2$.

An 𝑛×𝑚 matrix will have 𝑛 rows and 𝑚 columns:

$$\begin{bmatrix}
    x_{0,0} & x_{0,1} & \dotsb & x_{0,m-1} \\
    x_{1,0} & x_{1,1} & \dotsb & x_{1,m-1} \\
    \vdots  & \vdots  & \ddots & \vdots  \\
    x_{n-1,0} & x_{n-1,1} & \dotsb & x_{n-1,m-1}
\end{bmatrix}$$

A 1×1 matrix is equivalent to a scalar:
$$\begin{bmatrix} 3 \end{bmatrix} = 3$$
Quantum computing uses complex-valued matrices: the elements of a matrix can be complex numbers. This, for example, is a valid complex-valued matrix:$$\begin{bmatrix}
    1 & i \\
    -2i & 3 + 4i
\end{bmatrix}$$
Finally, a **vector** is an 𝑛×1 matrix. Here, for example, is a 3×1 vector:$$V = \begin{bmatrix} 1 \\ 2i \\ 3 + 4i \end{bmatrix}$$
Since vectors always have a width of 1, vector elements are sometimes written using only one index. In the above example, $V_0 = 1$ and $V_1 = 2i$.


### Matrix Addition
---

$$\begin{bmatrix}
    x_{0,0} & x_{0,1} & \dotsb & x_{0,m-1} \\
    x_{1,0} & x_{1,1} & \dotsb & x_{1,m-1} \\
    \vdots  & \vdots  & \ddots & \vdots  \\
    x_{n-1,0} & x_{n-1,1} & \dotsb & x_{n-1,m-1}
\end{bmatrix} +$$
$$+ \begin{bmatrix}
    y_{0,0} & y_{0,1} & \dotsb & y_{0,m-1} \\
    y_{1,0} & y_{1,1} & \dotsb & y_{1,m-1} \\
    \vdots  & \vdots  & \ddots & \vdots  \\
    y_{n-1,0} & y_{n-1,1} & \dotsb & y_{n-1,m-1}
\end{bmatrix} =$$
$$= \begin{bmatrix}
    x_{0,0} + y_{0,0} & x_{0,1} + y_{0,1} & \dotsb & x_{0,m-1} + y_{0,m-1} \\
    x_{1,0} + y_{1,0} & x_{1,1} + y_{1,1} & \dotsb & x_{1,m-1} + y_{1,m-1} \\
    \vdots  & \vdots  & \ddots & \vdots  \\
    x_{n-1,0} + y_{n-1,0} & x_{n-1,1} + y_{n-1,1} & \dotsb & x_{n-1,m-1} + y_{n-1,m-1}
\end{bmatrix}$$

Similarly, we can compute 𝐴−𝐵 by subtracting elements of 𝐵 from corresponding elements of 𝐴.

Matrix addition has the following properties:

- Commutativity: 𝐴+𝐵=𝐵+𝐴
- Associativity: (𝐴+𝐵)+𝐶=𝐴+(𝐵+𝐶)

### Scalar Multiplication
---

The next matrix operation is **scalar multiplication** - multiplying the entire matrix by a scalar (real or complex number):

$$a \cdot
\begin{bmatrix}
    x_{0,0} & x_{0,1} & \dotsb & x_{0,m-1} \\
    x_{1,0} & x_{1,1} & \dotsb & x_{1,m-1} \\
    \vdots  & \vdots  & \ddots & \vdots  \\
    x_{n-1,0} & x_{n-1,1} & \dotsb & x_{n-1,m-1}
\end{bmatrix} =
\begin{bmatrix}
    a \cdot x_{0,0} & a \cdot x_{0,1} & \dotsb & a \cdot x_{0,m-1} \\
    a \cdot x_{1,0} & a \cdot x_{1,1} & \dotsb & a \cdot x_{1,m-1} \\
    \vdots  & \vdots  & \ddots & \vdots  \\
    a \cdot x_{n-1,0} & a \cdot x_{n-1,1} & \dotsb & a \cdot x_{n-1,m-1}
\end{bmatrix}$$

Scalar multiplication has the following properties:

- Associativity: 𝑥⋅(𝑦𝐴)=(𝑥⋅𝑦)𝐴
- Distributivity over matrix addition: 𝑥(𝐴+𝐵)=𝑥𝐴+𝑥𝐵
- Distributivity over scalar addition:

### Matrix Multiplication
---

**Matrix multiplication** is a very important and somewhat unusual operation. The unusual thing about it is that neither its operands nor its output are the same size: an 𝑛×𝑚 matrix multiplied by an 𝑚×𝑘 matrix results in an 𝑛×𝑘 matrix. That is, for matrix multiplication to be applicable, the number of columns in the first matrix must equal the number of rows in the second matrix.

Here is how matrix product is calculated: if we are calculating 𝐴𝐵=𝐶, then

$$C_{i,j} = A_{i,0} \cdot B_{0,j} + A_{i,1} \cdot B_{1,j} + \dotsb + A_{i,m-1} \cdot B_{m-1,j} = \sum_{t = 0}^{m-1} A_{i,t} \cdot B_{t,j}$$

Here is a small example:

$$\begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6
\end{bmatrix}
\begin{bmatrix}
    1 \\
    2 \\
    3
\end{bmatrix} =
\begin{bmatrix}
    1 \cdot 1 + 2 \cdot 2 + 3 \cdot 3 \\
    4 \cdot 1 + 5 \cdot 2 + 6 \cdot 3
\end{bmatrix} =
\begin{bmatrix}
    14 \\
    32
\end{bmatrix}$$

Matrix multiplication has the following properties:

- Associativity: 𝐴(𝐵𝐶)=(𝐴𝐵)𝐶
- Distributivity over matrix addition: 𝐴(𝐵+𝐶)=𝐴𝐵+𝐴𝐶 and (𝐴+𝐵)𝐶=𝐴𝐶+𝐵𝐶
- Associativity with scalar multiplication: 𝑥𝐴𝐵=𝑥(𝐴𝐵)=𝐴(𝑥𝐵)

> Note that matrix multiplication is **not commutative:** 𝐴𝐵 rarely equals 𝐵𝐴.

Another very important property of matrix multiplication is that a matrix multiplied by a vector produces another vector.


### Identity Matrix
---

An **identity matrix** 𝐼𝑛 is a special 𝑛×𝑛 matrix which has 1s on the main diagonal, and 0s everywhere else:

$$I_n =
\begin{bmatrix}
    1 & 0 & \dotsb & 0 \\
    0 & 1 & \dotsb & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \dotsb & 1
\end{bmatrix}$$

What makes it special is that multiplying any matrix (of compatible size) by $I_n$ returns the original matrix. To put it another way, if 𝐴 is an 𝑛×𝑚 matrix:

$$AI_m = I_nA = A$$

This is why $I_n$ is called an identity matrix - it acts as a **multiplicative identity**. In other words, it is the matrix equivalent of the number 1.

### Inverse Matrices
---


A square 𝑛×𝑛 matrix 𝐴 is **invertible** if it has an inverse 𝑛×𝑛 matrix $A^{-1}$ with the following property:

$$AA^{-1} = A^{-1}A = I_n$$

In other words, $A^{-1}$ acts as the **multiplicative inverse** of 𝐴.

Another, equivalent definition highlights what makes this an interesting property. For any matrices 𝐵 and 𝐶 of compatible sizes:

$$A^{-1}(AB) = A(A^{-1}B) = B$$
$$(CA)A^{-1} = (CA^{-1})A = C$$

A square matrix has a property called the **determinant**, with the determinant of matrix 𝐴 being written as |𝐴|. A matrix is invertible if and only if its determinant isn't equal to 0.

For a 2×2 matrix 𝐴, the determinant is defined as $|A| = A_{0,0} \cdot A_{1,1} - A_{0,1} \cdot A_{1,0}$.

### Matrix Transposition
---

The **transpose** operation, denoted as $A^T$, is essentially a reflection of the matrix across the diagonal: $A^T_{i,j} = A_{j,i}$.

Given an 𝑛×𝑚 matrix 𝐴, its transpose is the 𝑚×𝑛 matrix $A^T$, such that if:

$$A =
\begin{bmatrix}
    x_{0,0} & x_{0,1} & \dotsb & x_{0,m-1} \\
    x_{1,0} & x_{1,1} & \dotsb & x_{1,m-1} \\
    \vdots & \vdots & \ddots & \vdots \\
    x_{n-1,0} & x_{n-1,1} & \dotsb & x_{n-1,m-1}
\end{bmatrix}$$

then:
$$A^T =
\begin{bmatrix}
    x_{0,0} & x_{1,0} & \dotsb & x_{n-1,0} \\
    x_{0,1} & x_{1,1} & \dotsb & x_{n-1,1} \\
    \vdots & \vdots & \ddots & \vdots \\
    x_{0,m-1} & x_{1,m-1} & \dotsb & x_{n-1,m-1}
\end{bmatrix}$$

For example:
$$\begin{bmatrix}
    1 & 2 \\
    3 & 4 \\
    5 & 6
\end{bmatrix}^T =
\begin{bmatrix}
    1 & 3 & 5 \\
    2 & 4 & 6
\end{bmatrix}$$
A **symmetric** matrix is a square matrix which equals its own transpose: $A = A^T$. To put it another way, it has reflection symmetry (hence the name) across the main diagonal. For example, the following matrix is symmetric

$$\begin{bmatrix}
    1 & 2 & 3 \\
    2 & 4 & 5 \\
    3 & 5 & 6
\end{bmatrix}$$

The transpose of a matrix product is equal to the product of transposed matrices, taken in reverse order:$$(AB)^T = B^TA^T$$

### Matrix Conjugates
---
The next important single-matrix operation is the **matrix conjugate**, denoted as 𝐴―. This operation makes sense only for complex-valued matrices; as the name might suggest, it involves taking the complex conjugate of every element of the matrix: if

$$A =
\begin{bmatrix}
    x_{0,0} & x_{0,1} & \dotsb & x_{0,m-1} \\
    x_{1,0} & x_{1,1} & \dotsb & x_{1,m-1} \\
    \vdots & \vdots & \ddots & \vdots \\
    x_{n-1,0} & x_{n-1,1} & \dotsb & x_{n-1,m-1}
\end{bmatrix}$$

Then:

$$\overline{A} =
\begin{bmatrix}
    \overline{x}_{0,0} & \overline{x}_{0,1} & \dotsb & \overline{x}_{0,m-1} \\
    \overline{x}_{1,0} & \overline{x}_{1,1} & \dotsb & \overline{x}_{1,m-1} \\
    \vdots & \vdots & \ddots & \vdots \\
    \overline{x}_{n-1,0} & \overline{x}_{n-1,1} & \dotsb & \overline{x}_{n-1,m-1}
\end{bmatrix}$$

> As a reminder, a conjugate of a complex number $x = a + bi$ is $\overline{x} = a - bi$

The conjugate of a matrix product equals to the product of conjugates of the matrices:

$$\overline{AB} = (\overline{A})(\overline{B})$$

### Adjoint Matrix 
---

The final important single-matrix operation is a combination of the previous two. The **conjugate transpose**, also called the **adjoint** of matrix 𝐴, is defined as $A^\dagger = \overline{(A^T)} = (\overline{A})^T$.

A matrix is known as **Hermitian** or **self-adjoint** if it equals its own adjoint: $A = A^\dagger$. For example, the following matrix is Hermitian:

$$\begin{bmatrix}
    1 & i \\
    -i & 2
\end{bmatrix}$$
The adjoint of a matrix product can be calculated as follows:

$$(AB)^\dagger = B^\dagger A^\dagger$$

### Unitary Matrices
---

**Unitary matrices** are very important for quantum computing. A matrix is unitary when it is invertible, and its inverse is equal to its adjoint: $U^{-1} = U^\dagger$. That is, an 𝑛×𝑛 square matrix 𝑈 is unitary if and only if $UU^\dagger = U^\dagger U = I_n$.

Is this matrix unitary?

$$A = \begin{bmatrix}
    \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
    \frac{i}{\sqrt{2}} & \frac{-i}{\sqrt{2}}
\end{bmatrix} = 
\frac{1}{\sqrt{2}} \begin{bmatrix}
    1 & 1 \\
    i & -i
\end{bmatrix}$$

To check whether the input matrix is unitary, we will need to perform the following steps:

1. Calculate the adjoint of the input matrix $A^\dagger$.

$$A^\dagger = \frac{1}{\sqrt{2}} \begin{bmatrix}
     1 & -i \\
     1 & i
 \end{bmatrix}$$

1. Multiply it by the input matrix.

$$AA^\dagger = \frac12 \begin{bmatrix}
     1 & 1 \\
     i & -i
 \end{bmatrix} \begin{bmatrix}
     1 & -i \\
     1 & i
 \end{bmatrix} = \frac12 \begin{bmatrix}
     1 \cdot 1 + 1 \cdot 1 & 1 \cdot (-i) + 1 \cdot i \\
     i \cdot 1 + (-i) \cdot 1 & i \cdot (-i) + (-i) \cdot i
 \end{bmatrix} = \begin{bmatrix}
     1 & 0 \\
     0 & 1
 \end{bmatrix}$$

If the multiplication result $AA^\dagger$ is an identity matrix, which is indeed the case, and the product $A^\dagger A$ is also an identity matrix (which you can verify in a similar manner), the matrix is unitary.


### Inner Product
---
The **inner product** is yet another important matrix operation that is only applied to vectors. Given two vectors 𝑉 and 𝑊 of the same size, their inner product $\langle V , W \rangle$ is defined as a product of matrices $V^\dagger$ and 𝑊:

$$\langle V , W \rangle = V^\dagger W$$

Let's break this down so it's a bit easier to understand. A 1×𝑛 matrix (the adjoint of an 𝑛×1 vector) multiplied by an 𝑛×1 vector results in a 1×1 matrix (which is equivalent to a scalar). The result of an inner product is that scalar.

That is, to calculate the inner product of two vectors, take the corresponding elements $V_k$ and $W_k$, multiply the complex conjugate of $V_k$ by $W_k$, and add up those products:

$$\langle V , W \rangle = \sum_{k=0}^{n-1}\overline{V_k}W_k$$

If you are familiar with the **dot product**, you will notice that it is equivalent to inner product for real-numbered vectors.

>We use our definition for these tutorials because it matches the notation used in quantum computing. You might encounter other sources which define the inner product a little differently: $\langle V , W \rangle = W^\dagger V = V^T\overline{W}$, in contrast to the $V^\dagger W$ that we use. These definitions are almost equivalent, with some differences in the scalar multiplication by a complex number.

#### Vector Norm
---

An immediate application for the inner product is computing the **vector norm**. The norm of vector 𝑉 is defined as $||V|| = \sqrt{\langle V , V \rangle}$. This condenses the vector down to a single non-negative real value. If the vector represents coordinates in space, the norm happens to be the length of the vector. A vector is called normalized if its norm is equal to 1.

The inner product has the following properties:

- Distributivity over addition: <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mi>V</mi>
  <mo>+</mo>
  <mi>W</mi>
  <mo>,</mo>
  <mi>X</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mi>V</mi>
  <mo>,</mo>
  <mi>X</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>+</mo>
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mi>W</mi>
  <mo>,</mo>
  <mi>X</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
</math> and <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mi>V</mi>
  <mo>,</mo>
  <mi>W</mi>
  <mo>+</mo>
  <mi>X</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mi>V</mi>
  <mo>,</mo>
  <mi>W</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>+</mo>
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mi>V</mi>
  <mo>,</mo>
  <mi>X</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
</math>
- Partial associativity with scalar multiplication: <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>x</mi>
  <mo>&#x22C5;</mo>
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mi>V</mi>
  <mo>,</mo>
  <mi>W</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mover>
    <mi>x</mi>
    <mo accent="true">&#x2015;</mo>
  </mover>
  <mi>V</mi>
  <mo>,</mo>
  <mi>W</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mi>V</mi>
  <mo>,</mo>
  <mi>x</mi>
  <mi>W</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
</math>
- Skew symmetry: <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mi>V</mi>
  <mo>,</mo>
  <mi>W</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mover>
    <mrow>
      <mo fence="false" stretchy="false">&#x27E8;</mo>
      <mi>W</mi>
      <mo>,</mo>
      <mi>V</mi>
      <mo fence="false" stretchy="false">&#x27E9;</mo>
    </mrow>
    <mo accent="true">&#x2015;</mo>
  </mover>
</math>
- Multiplying a vector by a unitary matrix **preserves the vector's inner product with itself** (and therefore the vector's norm): <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mi>U</mi>
  <mi>V</mi>
  <mo>,</mo>
  <mi>U</mi>
  <mi>V</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mi>V</mi>
  <mo>,</mo>
  <mi>V</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
</math>

> Note that just like matrix multiplication, the inner product is **not commutative**: ⟨𝑉,𝑊⟩ won't always equal ⟨𝑊,𝑉⟩.

#### Outer Products
---

The **outer product** of two vectors 𝑉 and 𝑊 is defined as 𝑉𝑊†. That is, the outer product of an 𝑛×1 vector and an 𝑚×1 vector is an 𝑛×𝑚 matrix. If we denote the outer product of 𝑉 and 𝑊 as 𝑋, then $X_{i,j} = V_i \cdot \overline{W_j}$.

### Tensor Products
---

The **tensor product** is a different way of multiplying matrices. Rather than multiplying rows by columns, the tensor product multiplies the second matrix by every element of the first matrix.

Given 𝑛×𝑚 matrix 𝐴 and 𝑘×𝑙 matrix 𝐵, their tensor product 𝐴⊗𝐵 is an (𝑛⋅𝑘)×(𝑚⋅𝑙) matrix defined as follows:

$$A \otimes B =
\begin{bmatrix}
    A_{0,0} \cdot B & A_{0,1} \cdot B & \dotsb & A_{0,m-1} \cdot B \\
    A_{1,0} \cdot B & A_{1,1} \cdot B & \dotsb & A_{1,m-1} \cdot B \\
    \vdots & \vdots & \ddots & \vdots \\
    A_{n-1,0} \cdot B & A_{n-1,1} \cdot B & \dotsb & A_{n-1,m-1} \cdot B
\end{bmatrix} =$$
$$= \begin{bmatrix}
    A_{0,0} \cdot \begin{bmatrix}B_{0,0} & \dotsb & B_{0,l-1} \\ \vdots & \ddots & \vdots \\ B_{k-1,0} & \dotsb & b_{k-1,l-1} \end{bmatrix} & \dotsb &
    A_{0,m-1} \cdot \begin{bmatrix}B_{0,0} & \dotsb & B_{0,l-1} \\ \vdots & \ddots & \vdots \\ B_{k-1,0} & \dotsb & B_{k-1,l-1} \end{bmatrix} \\
    \vdots & \ddots & \vdots \\
    A_{n-1,0} \cdot \begin{bmatrix}B_{0,0} & \dotsb & B_{0,l-1} \\ \vdots & \ddots & \vdots \\ B_{k-1,0} & \dotsb & B_{k-1,l-1} \end{bmatrix} & \dotsb &
    A_{n-1,m-1} \cdot \begin{bmatrix}B_{0,0} & \dotsb & B_{0,l-1} \\ \vdots & \ddots & \vdots \\ B_{k-1,0} & \dotsb & B_{k-1,l-1} \end{bmatrix}
\end{bmatrix} =$$
$$= \begin{bmatrix}
    A_{0,0} \cdot B_{0,0} & \dotsb & A_{0,0} \cdot B_{0,l-1} & \dotsb & A_{0,m-1} \cdot B_{0,0} & \dotsb & A_{0,m-1} \cdot B_{0,l-1} \\
    \vdots & \ddots & \vdots & \dotsb & \vdots & \ddots & \vdots \\
    A_{0,0} \cdot B_{k-1,0} & \dotsb & A_{0,0} \cdot B_{k-1,l-1} & \dotsb & A_{0,m-1} \cdot B_{k-1,0} & \dotsb & A_{0,m-1} \cdot B_{k-1,l-1} \\
    \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
    A_{n-1,0} \cdot B_{0,0} & \dotsb & A_{n-1,0} \cdot B_{0,l-1} & \dotsb & A_{n-1,m-1} \cdot B_{0,0} & \dotsb & A_{n-1,m-1} \cdot B_{0,l-1} \\
    \vdots & \ddots & \vdots & \dotsb & \vdots & \ddots & \vdots \\
    A_{n-1,0} \cdot B_{k-1,0} & \dotsb & A_{n-1,0} \cdot B_{k-1,l-1} & \dotsb & A_{n-1,m-1} \cdot B_{k-1,0} & \dotsb & A_{n-1,m-1} \cdot B_{k-1,l-1}
\end{bmatrix}$$

Notice that the tensor product of two vectors is another vector: if 𝑉 is an 𝑛×1 vector, and 𝑊 is an 𝑚×1 vector, 𝑉⊗𝑊 is an (𝑛⋅𝑚)×1 vector.

The tensor product has the following properties:

- Distributivity over addition: $(A + B) \otimes C = A \otimes C + B \otimes C$ , $A \otimes (B + C) = A \otimes B + A \otimes C$
- Associativity with scalar multiplication: $x(A \otimes B) = (xA) \otimes B = A \otimes (xB)$
- Mixed-product property (relation with matrix multiplication): $(A \otimes B) (C \otimes D) = (AC) \otimes (BD)$