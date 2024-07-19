
---
The basic building block of a classical computer is the bit - a single memory cell that is either in state 0 or in state 1. Similarly, the basic building block of a quantum computer is the quantum bit, or **qubit**. Like the classical bit, a qubit can be in state 0 or in state 1. Unlike the classical bit, however, the qubit isn't limited to just those two states - it may also be in a combination, or **superposition** of those states.

>A common misconception about quantum computing is that a qubit is always in state 1 or state 0, we just don't know which one until we "measure" it. That is not the case. A qubit in a superposition is in a linear combination of the states 0 and 1. When a qubit is measured, it is forced to collapse into one state or the other - in other words, measuring a qubit is an irreversible process that changes its initial state.


## Matrix Representation
---
The state of a qubit is represented by a complex vector of size 2:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mi>&#x3B1;</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>&#x3B2;</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
Here 𝛼 and 𝛽 are complex numbers. 𝛼 represents how "close" the qubit is to state 0, and 𝛽 represents how "close" the qubit is to state 1. 

This vector is normalized: <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>&#x3B1;</mi>
  <msup>
    <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
    <mn>2</mn>
  </msup>
  <mo>+</mo>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>&#x3B2;</mi>
  <msup>
    <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
    <mn>2</mn>
  </msup>
  <mo>=</mo>
  <mn>1</mn>
</math>

> How to calculate the [[Complex Numbers#Modulus|Complex Modulus]] here.

𝛼 and 𝛽 are known as the probability amplitudes of states 0 and 1, respectively.

## Basis States
---

A qubit in state 0 would be represented by the following vector:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
Likewise, a qubit in state 1 would be represented by this vector:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
Note that you can use scalar multiplication and vector addition to express any qubit state <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mi>&#x3B1;</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>&#x3B2;</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math> as a sum of these two vectors with certain weights 𝛼 and 𝛽, known as linear combination.
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mi>&#x3B1;</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>&#x3B2;</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>=</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mi>&#x3B1;</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>+</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>&#x3B2;</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>=</mo>
  <mi>&#x3B1;</mi>
  <mo>&#x22C5;</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>+</mo>
  <mi>&#x3B2;</mi>
  <mo>&#x22C5;</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
Because of this, qubit states 0 and 1 are known as basis states. These two vectors have two properties.

1. They are normalized.
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>,</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>,</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mn>1</mn>
</math>
2. They are orthogonal to each other.

	<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>,</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>,</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mn>0</mn>
</math>
> As a reminder, ⟨𝑉,𝑊⟩ is the inner product of 𝑉 and 𝑊.

This means that these vectors form an **orthonormal basis**. The basis of <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math> and <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math> is called the **computational basis**, also known as the **canonical basis**.

>There exist other orthonormal bases, for example, the **Hadamard basis**, formed by the vectors
><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mfrac>
            <mn>1</mn>
            <msqrt>
              <mn>2</mn>
            </msqrt>
          </mfrac>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mfrac>
            <mn>1</mn>
            <msqrt>
              <mn>2</mn>
            </msqrt>
          </mfrac>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mtext>&#xA0;and&#xA0;</mtext>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mfrac>
            <mn>1</mn>
            <msqrt>
              <mn>2</mn>
            </msqrt>
          </mfrac>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mo>&#x2212;</mo>
          <mfrac>
            <mn>1</mn>
            <msqrt>
              <mn>2</mn>
            </msqrt>
          </mfrac>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math> 
   You can check that these vectors are normalized, and orthogonal to each other. Any qubit state  can be expressed as a linear combination of these vectors:
   <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mi>&#x3B1;</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>&#x3B2;</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>&#x3B1;</mi>
      <mo>+</mo>
      <mi>&#x3B2;</mi>
    </mrow>
    <msqrt>
      <mn>2</mn>
    </msqrt>
  </mfrac>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mfrac>
            <mn>1</mn>
            <msqrt>
              <mn>2</mn>
            </msqrt>
          </mfrac>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mfrac>
            <mn>1</mn>
            <msqrt>
              <mn>2</mn>
            </msqrt>
          </mfrac>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>+</mo>
  <mfrac>
    <mrow>
      <mi>&#x3B1;</mi>
      <mo>&#x2212;</mo>
      <mi>&#x3B2;</mi>
    </mrow>
    <msqrt>
      <mn>2</mn>
    </msqrt>
  </mfrac>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mfrac>
            <mn>1</mn>
            <msqrt>
              <mn>2</mn>
            </msqrt>
          </mfrac>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mo>&#x2212;</mo>
          <mfrac>
            <mn>1</mn>
            <msqrt>
              <mn>2</mn>
            </msqrt>
          </mfrac>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
>The Hadamard basis is widely used in quantum computing, for example, in the [BB84 quantum key distribution protocol](https://en.wikipedia.org/wiki/BB84).


## The Dirac Notation
---

Dirac notation is a shorthand notation that eases writing quantum states and computing linear algebra. In Dirac notation, a vector is denoted by a symbol called a **ket**. For example, a qubit in state 0 is represented by the ket |0⟩, and a qubit in state 1 is represented by the ket |1⟩:

![[Pasted image 20240625200401.png]]

These two kets represent basis states, so they can be used to represent any other state:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mi>&#x3B1;</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>&#x3B2;</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>=</mo>
  <mi>&#x3B1;</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>0</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>+</mo>
  <mi>&#x3B2;</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
</math>
Dirac notation is not only restricted to vectors 0 and 1; it can be used to represent any vector, similar to how variable names are used in algebra. For example, we can call the state above "the state 𝜓" and write it as:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>&#x3C8;</mi>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>=</mo>
  <mi>&#x3B1;</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>0</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>+</mo>
  <mi>&#x3B2;</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
</math>
Several ket symbols have a generally accepted use, so you will see them often:

![[Pasted image 20240625200955.png]]

We will learn more about Dirac notation in the next katas, as we introduce quantum gates and multi-qubit systems.

## Relative and Global Phase
---

Complex numbers have a parameter called the phase. If a complex number 𝑧=𝑥+𝑖𝑦 is written in polar form <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>z</mi>
  <mo>=</mo>
  <mi>r</mi>
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>&#x3B8;</mi>
    </mrow>
  </msup>
</math>, its phase is 𝜃, where 𝜃=𝑎𝑡𝑎𝑛2(𝑦,𝑥).

>`atan2` is a useful function available in most programming languages. It takes two arguments and returns an angle 𝜃 between −𝜋 and 𝜋 that has cos⁡𝜃=𝑥 and sin⁡𝜃=𝑦. Unlike using <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>tan</mi>
    <mrow data-mjx-texclass="ORD">
      <mo>&#x2212;</mo>
      <mn>1</mn>
    </mrow>
  </msup>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mo stretchy="false">(</mo>
  <mfrac>
    <mi>y</mi>
    <mi>x</mi>
  </mfrac>
  <mo stretchy="false">)</mo>
</math>, `atan2` computes the correct quadrant for the angle, since it preserves information about the signs of both sine and cosine of the angle.

The probability amplitudes 𝛼 and 𝛽 are complex numbers, therefore 𝛼 and 𝛽 have a phase. For example, consider a qubit in state <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mfrac>
    <mrow>
      <mn>1</mn>
      <mo>+</mo>
      <mi>i</mi>
    </mrow>
    <mn>2</mn>
  </mfrac>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>0</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>+</mo>
  <mfrac>
    <mrow>
      <mn>1</mn>
      <mo>&#x2212;</mo>
      <mi>i</mi>
    </mrow>
    <mn>2</mn>
  </mfrac>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
</math>. If you do the math, you see that the phase of |0⟩ is
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>a</mi>
  <mi>t</mi>
  <mi>a</mi>
  <mi>n</mi>
  <mn>2</mn>
  <mo stretchy="false">(</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo>,</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mi>&#x3C0;</mi>
    <mn>4</mn>
  </mfrac>
</math> , and the phase of |1⟩ is <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>a</mi>
  <mi>t</mi>
  <mi>a</mi>
  <mi>n</mi>
  <mn>2</mn>
  <mo stretchy="false">(</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo>,</mo>
  <mo>&#x2212;</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mo>&#x2212;</mo>
  <mfrac>
    <mi>&#x3C0;</mi>
    <mn>4</mn>
  </mfrac>
</math>. The difference between these two phases is known as **relative phase**.

Multiplying the state of the entire system by <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>&#x3B8;</mi>
    </mrow>
  </msup>
</math> doesn't affect the relative phase: 𝛼|0⟩+𝛽|1⟩ has the same relative phase as <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>&#x3B8;</mi>
    </mrow>
  </msup>
</math>(𝛼|0⟩+𝛽|1⟩). In the second expression, 𝜃 is known as the system's **global phase**.

The state of a qubit (or, more generally, the state of a quantum system) is defined by its relative phase - global phase arises as a consequence of using linear algebra to represent qubits, and has no physical meaning. That is, applying a phase to the entire state of a system (multiplying the entire vector by <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>&#x3B8;</mi>
    </mrow>
  </msup>
</math> for any real 𝜃) doesn't actually affect the state of the system. Because of this, global phase is sometimes known as **unobservable phase** or **hidden phase**.

## Q# 
---

### Qubit Data Type
---

In Q#, qubits are represented by the `Qubit` data type. On a physical quantum computer, it's impossible to directly access the state of a qubit, whether to read its exact state, or to set it to a desired state, and this data type reflects that. Instead, you can change the state of a qubit using quantum gates, and extract information about the state of the system using measurements.

That being said, when you run Q# code on a quantum simulator instead of a physical quantum computer, you can use diagnostic functions that allow you to peek at the state of the quantum system. This is very useful both for learning and for debugging small Q# programs.

The qubits aren't an ordinary data type, so the variables of this type have to be declared and initialized ("allocated") a little differently. The `use` statement allocates a qubit (or multiple) that can be used until the end of the scope in which the statement was used: `use q = Qubit();` allocates a qubit and binds it to the variable `q`.

Freshly allocated qubits start out in state |0⟩, and have to be returned to that state by the time they are released. If you attempt to release a qubit in any state other than |0⟩, it will result in a runtime error. We will see why it is important later, when we look at multi-qubit systems.

### Visualizing Quantum State
---

Before we continue, let's learn some techniques to visualize the quantum state of our qubits.

#### Display the Quantum State of a Single-Qubit Program

Let's start with a simple scenario: a program that acts on a single qubit. The state of the quantum system used by this program can be represented as a complex vector of length 2, or, using Dirac notation,

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mi>&#x3B1;</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>&#x3B2;</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>=</mo>
  <mi>&#x3B1;</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>0</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>+</mo>
  <mi>&#x3B2;</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
</math>
If this program runs on a physical quantum system, there is no way to get the information about the values of 𝛼 and 𝛽 at a certain point of the program execution from a single observation. You would need to run the program repeatedly up to this point, perform a measurement on the system, and aggregate the results of multiple measurements to estimate 𝛼 and 𝛽.

However, at the early stages of quantum program development the program typically runs on a simulator - a classical program which simulates the behavior of a small quantum system while having complete information about its internal state. You can take advantage of this to do some non-physical things, such as peeking at the internals of the quantum system to observe its exact state without disturbing it!

The `DumpMachine` function from the `Microsoft.Quantum.Diagnostics` namespace allows you to do exactly that. The output of `DumpMachine` is accurate up to a global phase, and remember that global phase does not have any physical meaning. When using `DumpMachine`, you may see that all probability amplitudes are multiplied by some complex number compared to the state you're expecting.

#### Demo : DumpMachine For Single-Qubit Systems
---

The following demo shows how to allocate a qubit and examine its state in Q#. You'll use `DumpMachine` to output the state of the system at any point in the program without affecting the state.

>Note that the Q# code doesn't have access to the output of `DumpMachine`, so you cannot write any non-physical code in Q#!

```c#
namespace Kata {
    open Microsoft.Quantum.Diagnostics;

    @EntryPoint()
    operation RunExample() : Unit {
        // This line allocates a qubit in state |0⟩.
        use q = Qubit();
        Message("State |0⟩:");

        // This line prints out the state of the quantum computer.
        // Since only one qubit is allocated, only its state is printed.
        DumpMachine();

        // This line changes the qubit from state |0⟩ to state |1⟩.
        X(q);

        Message("State |1⟩:");
        DumpMachine();

        // This line changes the qubit to state |-⟩ = (1/sqrt(2))(|0⟩ - |1⟩).
        // That is, this puts the qubit into a superposition where the absolute
        // value of the probability amplitudes is 1/sqrt(2), which is
        // approximately 0.707107.
        H(q);

        Message("State |-⟩:");
        DumpMachine();

        // This line changes the qubit to state |-i⟩ = (1/sqrt(2))(|0⟩ - i|1⟩).
        S(q);

        Message("State |-i⟩:");
        DumpMachine();

        // This will put the qubit into an uneven superposition, where the
        // amplitudes of |0⟩ and |1⟩ have different absolute values.
        Rx(2.0, q);
        Ry(1.0, q);

        Message("Uneven superposition state:");
        DumpMachine();

        // This line returns the qubit to state |0⟩, which must be done before
        // the qubit is released or otherwise a runtime error might occur.
        Reset(q);
    }
}

```

>It is important to note that although we reason about quantum systems in terms of their state, Q# does not have any representation of the quantum state in the language. Instead, state is an internal property of the quantum system, modified using gates. For more information, see [Q# documentation on quantum states](https://learn.microsoft.com/azure/quantum/concepts-dirac-notation#q-gate-sequences-equivalent-to-quantum-states).