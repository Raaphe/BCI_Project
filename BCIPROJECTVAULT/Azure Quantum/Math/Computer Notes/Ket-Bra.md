
---

## Ket-Bra Representation
---

There is another way to represent quantum gates, this time using Dirac notation. However, the kets we've been using aren't enough to represent arbitrary matrices. We need to introduce another piece of notation: the **bra** (this is why Dirac notation is sometimes called **bra-ket notation**).

Recall that kets represent column vectors; a bra is a ket's row vector counterpart. For any ket |𝜓⟩, the corresponding bra is its adjoint (conjugate transpose): <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow data-mjx-texclass="ORD">
    <mo fence="false" stretchy="false">&#x27E8;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>&#x3C8;</mi>
    </mrow>
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
  </mrow>
  <mo>=</mo>
  <msup>
    <mrow data-mjx-texclass="ORD">
      <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
      <mrow data-mjx-texclass="ORD">
        <mi>&#x3C8;</mi>
      </mrow>
      <mo fence="false" stretchy="false">&#x27E9;</mo>
    </mrow>
    <mo>&#x2020;</mo>
  </msup>
</math>.

Some examples:
![[Pasted image 20240629164559.png]]


> Reminder: [[Matrices#Adjoint Matrix |What is an Adjoint?]] 

Kets and bras give us a neat way to express inner and outer products. The inner product of |𝜙⟩ and |𝜓⟩ is the matrix product of ⟨𝜙| and |𝜓⟩, denoted as ⟨𝜙|𝜓⟩, and their outer product is the matrix product of |𝜙⟩ and ⟨𝜓|, denoted as |𝜙⟩⟨𝜓|. Notice that the norm of |𝜓⟩ is <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msqrt>
    <mrow data-mjx-texclass="INNER">
      <mo data-mjx-texclass="OPEN" stretchy="false">&#x27E8;</mo>
      <mi>&#x3C8;</mi>
      <mo stretchy="false" braketbar="true">|</mo>
      <mi>&#x3C8;</mi>
      <mo data-mjx-texclass="CLOSE" stretchy="false">&#x27E9;</mo>
    </mrow>
  </msqrt>
</math>.

This brings us to representing matrices. Recall that the [[Matrices#Outer Products|outer product]] of two vectors of the same size produces a square matrix. We can use a linear combination of several outer products of simple vectors (such as basis vectors) to express any square matrix. For example, the 𝑋 gate can be expressed as follows:

$$X = \ket{0}\bra{1} + \ket{1}\bra{0}$$
$$\ket{0}\bra{1} + \ket{1}\bra{0} =
\begin{bmatrix} 1 \\ 0 \end{bmatrix}\begin{bmatrix} 0 & 1 \end{bmatrix} +
\begin{bmatrix} 0 \\ 1 \end{bmatrix}\begin{bmatrix} 1 & 0 \end{bmatrix} =
\begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} + \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix} =
\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$

This representation can be used to carry out calculations in Dirac notation without ever switching back to matrix representation:

$$X\ket{0} = \big(\ket{0}\bra{1} + \ket{1}\bra{0}\big)\ket{0} = \ket{0}\braket{1|0} + \ket{1}\braket{0|0} = \ket{0}\big(\braket{1|0}\big) + \ket{1}\big(\braket{0|0}\big) = \ket{0}(0) + \ket{1}(1) = \ket{1}$$

>That last step may seem a bit confusing. Recall that |0⟩ and |1⟩ form an **orthonormal basis**. That is, they are both normalized, and they are orthogonal to each other. 
  >
   A vector is **normalized** if its norm is equal to 1, which only happens if its inner  product with itself is equal to 1. This means that ⟨0|0⟩=⟨1|1⟩=1
>
>Two vectors are **orthogonal** to each other if their [[Matrices#Inner Product|inner product]] equals 0. This means that ⟨0|1⟩=⟨1|0⟩=0.

In general case, a matrix

$$A = \begin{bmatrix} a_{00} & a_{01} \\ a_{10} & a_{11} \end{bmatrix}$$

will have the following ket-bra representation:

$$A = a_{00} \ket{0}\bra{0} + a_{01} \ket{0}\bra{1} + a_{10} \ket{1}\bra{0} + a_{11} \ket{1}\bra{1}$$
## Ket-Bra Decomposition
---
This section describes a more formal process of finding the ket-bra decompositions of quantum gates. This section is not necessary to start working with quantum gates, so feel free to skip it for now, and come back to it later and revisit the [[Qubit Gates#Pauli Gates|qubit gates section]]. 

You can use the properties of _eigenvalues_ and _eigenvectors_ to find the ket-bra decomposition of any gate. Given a gate 𝐴 and the orthogonal vectors |𝜙⟩ and |𝜓⟩, if:

$$A\ket{\phi} = x_\phi\ket{\phi}$$
$$A\ket{\psi} = x_\psi\ket{\psi}$$

Real numbers $x_\phi$ and $x_\psi$ are called eigenvalues and |𝜙⟩ and |𝜓⟩ are eigenvectors of 𝐴. Then:

$$A = x_\phi\ket{\phi}\bra{\phi} + x_\psi\ket{\psi}\bra{\psi}$$

Let's use our 𝑋 gate as a simple example. The 𝑋 gate has two eigenvectors: $\ket{+} = \frac{1}{\sqrt{2}}\big(\ket{0} + \ket{1}\big)$ and $\ket{-} = \frac{1}{\sqrt{2}}\big(\ket{0} - \ket{1}\big)$. Their eigenvalues are 1 and −1 respectively:

$$X\ket{+} = \ket{+}$$
$$X\ket{-} = -\ket{-}$$

Here's what the decomposition looks like:

$$X = \ket{+}\bra{+} - \ket{-}\bra{-} =$$
$$= \frac{1}{2}\big[\big(\ket{0} + \ket{1}\big)\big(\bra{0} + \bra{1}\big) - \big(\ket{0} - \ket{1}\big)\big(\bra{0} - \bra{1}\big)\big] =$$
$$= \frac{1}{2}\big(\ket{0}\bra{0} + \ket{0}\bra{1} + \ket{1}\bra{0} + \ket{1}\bra{1} - \ket{0}\bra{0} + \ket{0}\bra{1} + \ket{1}\bra{0} - \ket{1}\bra{1}\big) =$$
$$= \frac{1}{2}\big(2\ket{0}\bra{1} + 2\ket{1}\bra{0}\big) =$$
$$= \ket{0}\bra{1} + \ket{1}\bra{0}$$


>revisit the [[Qubit Gates#Pauli Gates|qubit gates section]].  