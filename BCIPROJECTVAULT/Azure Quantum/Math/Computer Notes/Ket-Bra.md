
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
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>X</mi>
  <mo>=</mo>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>0</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mo fence="false" stretchy="false">&#x27E8;</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
  </mrow>
  <mo>+</mo>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mo fence="false" stretchy="false">&#x27E8;</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>0</mn>
    </mrow>
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
  </mrow>
</math>
![[Pasted image 20240701145743.png]]

This representation can be used to carry out calculations in Dirac notation without ever switching back to matrix representation:

![[Pasted image 20240701145932.png]]

>That last step may seem a bit confusing. Recall that |0⟩ and |1⟩ form an **orthonormal basis**. That is, they are both normalized, and they are orthogonal to each other. 
  >
   A vector is **normalized** if its norm is equal to 1, which only happens if its inner  product with itself is equal to 1. This means that ⟨0|0⟩=⟨1|1⟩=1
>
>Two vectors are **orthogonal** to each other if their [[Matrices#Inner Product|inner product]] equals 0. This means that ⟨0|1⟩=⟨1|0⟩=0.

In general case, a matrix
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>A</mi>
  <mo>=</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <msub>
            <mi>a</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>00</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>a</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>01</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>a</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>10</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>a</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>11</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
will have the following ket-bra representation:

![[Pasted image 20240701150736.png]]
## Ket-Bra Decomposition
---
This section describes a more formal process of finding the ket-bra decompositions of quantum gates. This section is not necessary to start working with quantum gates, so feel free to skip it for now, and come back to it later and revisit the [[Qubit Gates#Pauli Gates|qubit gates section]]. 

You can use the properties of _eigenvalues_ and _eigenvectors_ to find the ket-bra decomposition of any gate. Given a gate 𝐴 and the orthogonal vectors |𝜙⟩ and |𝜓⟩, if:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>A</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>&#x3D5;</mi>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>=</mo>
  <msub>
    <mi>x</mi>
    <mi>&#x3D5;</mi>
  </msub>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>&#x3D5;</mi>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>A</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>&#x3C8;</mi>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>=</mo>
  <msub>
    <mi>x</mi>
    <mi>&#x3C8;</mi>
  </msub>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>&#x3C8;</mi>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
</math>
Real numbers <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>x</mi>
    <mi>&#x3D5;</mi>
  </msub>
</math> and <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>x</mi>
    <mi>&#x3C8;</mi>
  </msub>
</math> are called eigenvalues and |𝜙⟩ and |𝜓⟩ are eigenvectors of 𝐴. Then:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>A</mi>
  <mo>=</mo>
  <msub>
    <mi>x</mi>
    <mi>&#x3D5;</mi>
  </msub>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>&#x3D5;</mi>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mo fence="false" stretchy="false">&#x27E8;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>&#x3D5;</mi>
    </mrow>
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
  </mrow>
  <mo>+</mo>
  <msub>
    <mi>x</mi>
    <mi>&#x3C8;</mi>
  </msub>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>&#x3C8;</mi>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mo fence="false" stretchy="false">&#x27E8;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>&#x3C8;</mi>
    </mrow>
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
  </mrow>
</math>
Let's use our 𝑋 gate as a simple example. The 𝑋 gate has two eigenvectors: <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mo>+</mo>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <msqrt>
      <mn>2</mn>
    </msqrt>
  </mfrac>
  <mrow data-mjx-texclass="ORD">
    <mo minsize="1.2em" maxsize="1.2em">(</mo>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>0</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>+</mo>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mo minsize="1.2em" maxsize="1.2em">)</mo>
  </mrow>
</math> and <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mo>&#x2212;</mo>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <msqrt>
      <mn>2</mn>
    </msqrt>
  </mfrac>
  <mrow data-mjx-texclass="ORD">
    <mo minsize="1.2em" maxsize="1.2em">(</mo>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>0</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>&#x2212;</mo>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mo minsize="1.2em" maxsize="1.2em">)</mo>
  </mrow>
</math>. Their eigenvalues are 1 and −1 respectively:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>X</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mo>+</mo>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>=</mo>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mo>+</mo>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>X</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mo>&#x2212;</mo>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>=</mo>
  <mo>&#x2212;</mo>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mo>&#x2212;</mo>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
</math>

Here's what the decomposition looks like:
![[Pasted image 20240701153857.png]]

>revisit the [[Qubit Gates#Pauli Gates|qubit gates section]].  