
---


> Most of these notes are provided by Microsoft [here](https://quantum.microsoft.com/en-us/experience/quantum-katas) 
> Another good source that details much found on here can be found on [this page](https://en.wikipedia.org/wiki/Determinant)

A **matrix** is set of numbers arranged in a rectangular grid. Here is a 2 by 2 matrix:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>A</mi>
  <mo>=</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mn>2</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>3</mn>
        </mtd>
        <mtd>
          <mn>4</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>  
𝐴𝑖,𝑗 refers to the element in row 𝑖 and column 𝑗 of matrix 𝐴 (all indices are 0-based). In the above example, 𝐴0,1=2.

An 𝑛×𝑚 matrix will have 𝑛 rows and 𝑚 columns:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mo>&#x22F1;</mo>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>

A 1×1 matrix is equivalent to a scalar:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>3</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>=</mo>
  <mn>3</mn>
</math>
Quantum computing uses complex-valued matrices: the elements of a matrix can be complex numbers. This, for example, is a valid complex-valued matrix:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mi>i</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mo>&#x2212;</mo>
          <mn>2</mn>
          <mi>i</mi>
        </mtd>
        <mtd>
          <mn>3</mn>
          <mo>+</mo>
          <mn>4</mn>
          <mi>i</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
Finally, a **vector** is an 𝑛×1 matrix. Here, for example, is a 3×1 vector:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>V</mi>
  <mo>=</mo>
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
          <mn>2</mn>
          <mi>i</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>3</mn>
          <mo>+</mo>
          <mn>4</mn>
          <mi>i</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
Since vectors always have a width of 1, vector elements are sometimes written using only one index. In the above example, 𝑉0=1 and 𝑉1=2𝑖.


### Matrix Addition
---
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mo>&#x22F1;</mo>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>+</mo>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo>+</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mo>&#x22F1;</mo>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>=</mo>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo>=</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
          <mo>+</mo>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
          <mo>+</mo>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
          <mo>+</mo>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
          <mo>+</mo>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
          <mo>+</mo>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
          <mo>+</mo>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mo>&#x22F1;</mo>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
          <mo>+</mo>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
          <mo>+</mo>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
          <mo>+</mo>
          <msub>
            <mi>y</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
Similarly, we can compute 𝐴−𝐵 by subtracting elements of 𝐵 from corresponding elements of 𝐴.

Matrix addition has the following properties:

- Commutativity: 𝐴+𝐵=𝐵+𝐴
- Associativity: (𝐴+𝐵)+𝐶=𝐴+(𝐵+𝐶)

### Scalar Multiplication
---

The next matrix operation is **scalar multiplication** - multiplying the entire matrix by a scalar (real or complex number):

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>a</mi>
  <mo>&#x22C5;</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mo>&#x22F1;</mo>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
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
          <mi>a</mi>
          <mo>&#x22C5;</mo>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mi>a</mi>
          <mo>&#x22C5;</mo>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <mi>a</mi>
          <mo>&#x22C5;</mo>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>a</mi>
          <mo>&#x22C5;</mo>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mi>a</mi>
          <mo>&#x22C5;</mo>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <mi>a</mi>
          <mo>&#x22C5;</mo>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mo>&#x22F1;</mo>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>a</mi>
          <mo>&#x22C5;</mo>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mi>a</mi>
          <mo>&#x22C5;</mo>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <mi>a</mi>
          <mo>&#x22C5;</mo>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
Scalar multiplication has the following properties:

- Associativity: 𝑥⋅(𝑦𝐴)=(𝑥⋅𝑦)𝐴
- Distributivity over matrix addition: 𝑥(𝐴+𝐵)=𝑥𝐴+𝑥𝐵
- Distributivity over scalar addition:

### Matrix Multiplication
---

**Matrix multiplication** is a very important and somewhat unusual operation. The unusual thing about it is that neither its operands nor its output are the same size: an 𝑛×𝑚 matrix multiplied by an 𝑚×𝑘 matrix results in an 𝑛×𝑘 matrix. That is, for matrix multiplication to be applicable, the number of columns in the first matrix must equal the number of rows in the second matrix.

Here is how matrix product is calculated: if we are calculating 𝐴𝐵=𝐶, then
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mi>C</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>,</mo>
      <mi>j</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <msub>
    <mi>A</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>,</mo>
      <mn>0</mn>
    </mrow>
  </msub>
  <mo>&#x22C5;</mo>
  <msub>
    <mi>B</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>0</mn>
      <mo>,</mo>
      <mi>j</mi>
    </mrow>
  </msub>
  <mo>+</mo>
  <msub>
    <mi>A</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>,</mo>
      <mn>1</mn>
    </mrow>
  </msub>
  <mo>&#x22C5;</mo>
  <msub>
    <mi>B</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
      <mo>,</mo>
      <mi>j</mi>
    </mrow>
  </msub>
  <mo>+</mo>
  <mo>&#x22EF;</mo>
  <mo>+</mo>
  <msub>
    <mi>A</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>,</mo>
      <mi>m</mi>
      <mo>&#x2212;</mo>
      <mn>1</mn>
    </mrow>
  </msub>
  <mo>&#x22C5;</mo>
  <msub>
    <mi>B</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>m</mi>
      <mo>&#x2212;</mo>
      <mn>1</mn>
      <mo>,</mo>
      <mi>j</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <munderover>
    <mo data-mjx-texclass="OP">&#x2211;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>t</mi>
      <mo>=</mo>
      <mn>0</mn>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>m</mi>
      <mo>&#x2212;</mo>
      <mn>1</mn>
    </mrow>
  </munderover>
  <msub>
    <mi>A</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>,</mo>
      <mi>t</mi>
    </mrow>
  </msub>
  <mo>&#x22C5;</mo>
  <msub>
    <mi>B</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>t</mi>
      <mo>,</mo>
      <mi>j</mi>
    </mrow>
  </msub>
</math>
Here is a small example:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mn>2</mn>
        </mtd>
        <mtd>
          <mn>3</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>4</mn>
        </mtd>
        <mtd>
          <mn>5</mn>
        </mtd>
        <mtd>
          <mn>6</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
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
          <mn>2</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>3</mn>
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
          <mn>1</mn>
          <mo>&#x22C5;</mo>
          <mn>1</mn>
          <mo>+</mo>
          <mn>2</mn>
          <mo>&#x22C5;</mo>
          <mn>2</mn>
          <mo>+</mo>
          <mn>3</mn>
          <mo>&#x22C5;</mo>
          <mn>3</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>4</mn>
          <mo>&#x22C5;</mo>
          <mn>1</mn>
          <mo>+</mo>
          <mn>5</mn>
          <mo>&#x22C5;</mo>
          <mn>2</mn>
          <mo>+</mo>
          <mn>6</mn>
          <mo>&#x22C5;</mo>
          <mn>3</mn>
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
          <mn>14</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>32</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
Matrix multiplication has the following properties:

- Associativity: 𝐴(𝐵𝐶)=(𝐴𝐵)𝐶
- Distributivity over matrix addition: 𝐴(𝐵+𝐶)=𝐴𝐵+𝐴𝐶 and (𝐴+𝐵)𝐶=𝐴𝐶+𝐵𝐶
- Associativity with scalar multiplication: 𝑥𝐴𝐵=𝑥(𝐴𝐵)=𝐴(𝑥𝐵)

> Note that matrix multiplication is **not commutative:** 𝐴𝐵 rarely equals 𝐵𝐴.

Another very important property of matrix multiplication is that a matrix multiplied by a vector produces another vector.


### Identity Matrix
---

An **identity matrix** 𝐼𝑛 is a special 𝑛×𝑛 matrix which has 1s on the main diagonal, and 0s everywhere else:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mi>I</mi>
    <mi>n</mi>
  </msub>
  <mo>=</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mn>0</mn>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mo>&#x22F1;</mo>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
        <mtd>
          <mn>0</mn>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
What makes it special is that multiplying any matrix (of compatible size) by 𝐼𝑛 returns the original matrix. To put it another way, if 𝐴 is an 𝑛×𝑚 matrix:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>A</mi>
  <msub>
    <mi>I</mi>
    <mi>m</mi>
  </msub>
  <mo>=</mo>
  <msub>
    <mi>I</mi>
    <mi>n</mi>
  </msub>
  <mi>A</mi>
  <mo>=</mo>
  <mi>A</mi>
</math>
This is why 𝐼𝑛 is called an identity matrix - it acts as a **multiplicative identity**. In other words, it is the matrix equivalent of the number 1.

### Inverse Matrices
---


A square 𝑛×𝑛 matrix 𝐴 is **invertible** if it has an inverse 𝑛×𝑛 matrix 𝐴−1 with the following property:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>A</mi>
  <msup>
    <mi>A</mi>
    <mrow data-mjx-texclass="ORD">
      <mo>&#x2212;</mo>
      <mn>1</mn>
    </mrow>
  </msup>
  <mo>=</mo>
  <msup>
    <mi>A</mi>
    <mrow data-mjx-texclass="ORD">
      <mo>&#x2212;</mo>
      <mn>1</mn>
    </mrow>
  </msup>
  <mi>A</mi>
  <mo>=</mo>
  <msub>
    <mi>I</mi>
    <mi>n</mi>
  </msub>
</math>
In other words, 𝐴−1 acts as the **multiplicative inverse** of 𝐴.

Another, equivalent definition highlights what makes this an interesting property. For any matrices 𝐵 and 𝐶 of compatible sizes:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>A</mi>
    <mrow data-mjx-texclass="ORD">
      <mo>&#x2212;</mo>
      <mn>1</mn>
    </mrow>
  </msup>
  <mo stretchy="false">(</mo>
  <mi>A</mi>
  <mi>B</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mi>A</mi>
  <mo stretchy="false">(</mo>
  <msup>
    <mi>A</mi>
    <mrow data-mjx-texclass="ORD">
      <mo>&#x2212;</mo>
      <mn>1</mn>
    </mrow>
  </msup>
  <mi>B</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mi>B</mi>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo stretchy="false">(</mo>
  <mi>C</mi>
  <mi>A</mi>
  <mo stretchy="false">)</mo>
  <msup>
    <mi>A</mi>
    <mrow data-mjx-texclass="ORD">
      <mo>&#x2212;</mo>
      <mn>1</mn>
    </mrow>
  </msup>
  <mo>=</mo>
  <mo stretchy="false">(</mo>
  <mi>C</mi>
  <msup>
    <mi>A</mi>
    <mrow data-mjx-texclass="ORD">
      <mo>&#x2212;</mo>
      <mn>1</mn>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
  <mi>A</mi>
  <mo>=</mo>
  <mi>C</mi>
</math>
A square matrix has a property called the **determinant**, with the determinant of matrix 𝐴 being written as |𝐴|. A matrix is invertible if and only if its determinant isn't equal to 0.

For a 2×2 matrix 𝐴, the determinant is defined as |𝐴|=𝐴0,0⋅𝐴1,1−𝐴0,1⋅𝐴1,0.

### Matrix Transposition
---

The **transpose** operation, denoted as 𝐴𝑇, is essentially a reflection of the matrix across the diagonal: 𝐴𝑖,𝑗𝑇=𝐴𝑗,𝑖.

Given an 𝑛×𝑚 matrix 𝐴, its transpose is the 𝑚×𝑛 matrix 𝐴𝑇, such that if:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>A</mi>
  <mo>=</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mo>&#x22F1;</mo>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>then:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>A</mi>
    <mi>T</mi>
  </msup>
  <mo>=</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mo>&#x22F1;</mo>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>For example:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mrow data-mjx-texclass="INNER">
      <mo data-mjx-texclass="OPEN">[</mo>
      <mtable columnspacing="1em" rowspacing="4pt">
        <mtr>
          <mtd>
            <mn>1</mn>
          </mtd>
          <mtd>
            <mn>2</mn>
          </mtd>
        </mtr>
        <mtr>
          <mtd>
            <mn>3</mn>
          </mtd>
          <mtd>
            <mn>4</mn>
          </mtd>
        </mtr>
        <mtr>
          <mtd>
            <mn>5</mn>
          </mtd>
          <mtd>
            <mn>6</mn>
          </mtd>
        </mtr>
      </mtable>
      <mo data-mjx-texclass="CLOSE">]</mo>
    </mrow>
    <mi>T</mi>
  </msup>
  <mo>=</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mn>3</mn>
        </mtd>
        <mtd>
          <mn>5</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>2</mn>
        </mtd>
        <mtd>
          <mn>4</mn>
        </mtd>
        <mtd>
          <mn>6</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
A **symmetric** matrix is a square matrix which equals its own transpose: 𝐴=𝐴𝑇. To put it another way, it has reflection symmetry (hence the name) across the main diagonal. For example, the following matrix is symmetric:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mn>2</mn>
        </mtd>
        <mtd>
          <mn>3</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>2</mn>
        </mtd>
        <mtd>
          <mn>4</mn>
        </mtd>
        <mtd>
          <mn>5</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>3</mn>
        </mtd>
        <mtd>
          <mn>5</mn>
        </mtd>
        <mtd>
          <mn>6</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
The transpose of a matrix product is equal to the product of transposed matrices, taken in reverse order:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo stretchy="false">(</mo>
  <mi>A</mi>
  <mi>B</mi>
  <msup>
    <mo stretchy="false">)</mo>
    <mi>T</mi>
  </msup>
  <mo>=</mo>
  <msup>
    <mi>B</mi>
    <mi>T</mi>
  </msup>
  <msup>
    <mi>A</mi>
    <mi>T</mi>
  </msup>
</math>

### Matrix Conjugates
---
The next important single-matrix operation is the **matrix conjugate**, denoted as 𝐴―. This operation makes sense only for complex-valued matrices; as the name might suggest, it involves taking the complex conjugate of every element of the matrix: if

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>A</mi>
  <mo>=</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mo>&#x22F1;</mo>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mi>x</mi>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
Then:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mover>
    <mi>A</mi>
    <mo accent="true">&#x2015;</mo>
  </mover>
  <mo>=</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <msub>
            <mover>
              <mi>x</mi>
              <mo accent="true">&#x2015;</mo>
            </mover>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mover>
              <mi>x</mi>
              <mo accent="true">&#x2015;</mo>
            </mover>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mover>
              <mi>x</mi>
              <mo accent="true">&#x2015;</mo>
            </mover>
            <mrow data-mjx-texclass="ORD">
              <mn>0</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mover>
              <mi>x</mi>
              <mo accent="true">&#x2015;</mo>
            </mover>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mover>
              <mi>x</mi>
              <mo accent="true">&#x2015;</mo>
            </mover>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mover>
              <mi>x</mi>
              <mo accent="true">&#x2015;</mo>
            </mover>
            <mrow data-mjx-texclass="ORD">
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
        <mtd>
          <mo>&#x22F1;</mo>
        </mtd>
        <mtd>
          <mrow data-mjx-texclass="ORD">
            <mo>&#x22EE;</mo>
          </mrow>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mover>
              <mi>x</mi>
              <mo accent="true">&#x2015;</mo>
            </mover>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>0</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <msub>
            <mover>
              <mi>x</mi>
              <mo accent="true">&#x2015;</mo>
            </mover>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
        <mtd>
          <mo>&#x22EF;</mo>
        </mtd>
        <mtd>
          <msub>
            <mover>
              <mi>x</mi>
              <mo accent="true">&#x2015;</mo>
            </mover>
            <mrow data-mjx-texclass="ORD">
              <mi>n</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
              <mo>,</mo>
              <mi>m</mi>
              <mo>&#x2212;</mo>
              <mn>1</mn>
            </mrow>
          </msub>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
> As a reminder, a conjugate of a complex number <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>x</mi>
  <mo>=</mo>
  <mi>a</mi>
  <mo>+</mo>
  <mi>b</mi>
  <mi>i</mi>
</math> is <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mover>
    <mi>x</mi>
    <mo accent="true">&#x2015;</mo>
  </mover>
  <mo>=</mo>
  <mi>a</mi>
  <mo>&#x2212;</mo>
  <mi>b</mi>
  <mi>i</mi>
</math>.

The conjugate of a matrix product equals to the product of conjugates of the matrices:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mover>
    <mrow>
      <mi>A</mi>
      <mi>B</mi>
    </mrow>
    <mo accent="true">&#x2015;</mo>
  </mover>
  <mo>=</mo>
  <mo stretchy="false">(</mo>
  <mover>
    <mi>A</mi>
    <mo accent="true">&#x2015;</mo>
  </mover>
  <mo stretchy="false">)</mo>
  <mo stretchy="false">(</mo>
  <mover>
    <mi>B</mi>
    <mo accent="true">&#x2015;</mo>
  </mover>
  <mo stretchy="false">)</mo>
</math>

### Adjoint Matrix 
---

The final important single-matrix operation is a combination of the previous two. The **conjugate transpose**, also called the **adjoint** of matrix 𝐴, is defined as <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>A</mi>
    <mo>&#x2020;</mo>
  </msup>
  <mo>=</mo>
  <mover>
    <mrow>
      <mo stretchy="false">(</mo>
      <msup>
        <mi>A</mi>
        <mi>T</mi>
      </msup>
      <mo stretchy="false">)</mo>
    </mrow>
    <mo accent="true">&#x2015;</mo>
  </mover>
  <mo>=</mo>
  <mo stretchy="false">(</mo>
  <mover>
    <mi>A</mi>
    <mo accent="true">&#x2015;</mo>
  </mover>
  <msup>
    <mo stretchy="false">)</mo>
    <mi>T</mi>
  </msup>
</math>.

A matrix is known as **Hermitian** or **self-adjoint** if it equals its own adjoint: <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>A</mi>
  <mo>=</mo>
  <msup>
    <mi>A</mi>
    <mo>&#x2020;</mo>
  </msup>
</math>. For example, the following matrix is Hermitian:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mi>i</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mo>&#x2212;</mo>
          <mi>i</mi>
        </mtd>
        <mtd>
          <mn>2</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
### Unitary Matrices
---



**Unitary matrices** are very important for quantum computing. A matrix is unitary when it is invertible, and its inverse is equal to its adjoint: 𝑈−1=𝑈†. That is, an 𝑛×𝑛 square matrix 𝑈 is unitary if and only if <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>U</mi>
  <msup>
    <mi>U</mi>
    <mo>&#x2020;</mo>
  </msup>
  <mo>=</mo>
  <msup>
    <mi>U</mi>
    <mo>&#x2020;</mo>
  </msup>
  <mi>U</mi>
  <mo>=</mo>
  <msub>
    <mi>I</mi>
    <mi>n</mi>
  </msub>
</math>.

ex:

Is this matrix unitary?

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>A</mi>
  <mo>=</mo>
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
            <mi>i</mi>
            <msqrt>
              <mn>2</mn>
            </msqrt>
          </mfrac>
        </mtd>
        <mtd>
          <mfrac>
            <mrow>
              <mo>&#x2212;</mo>
              <mi>i</mi>
            </mrow>
            <msqrt>
              <mn>2</mn>
            </msqrt>
          </mfrac>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <msqrt>
      <mn>2</mn>
    </msqrt>
  </mfrac>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>i</mi>
        </mtd>
        <mtd>
          <mo>&#x2212;</mo>
          <mi>i</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
To check whether the input matrix is unitary, we will need to perform the following steps:

1. Calculate the adjoint of the input matrix 𝐴†.

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>A</mi>
    <mo>&#x2020;</mo>
  </msup>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <msqrt>
      <mn>2</mn>
    </msqrt>
  </mfrac>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mo>&#x2212;</mo>
          <mi>i</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mi>i</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
2. Multiply it by the input matrix.
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>A</mi>
  <msup>
    <mi>A</mi>
    <mo>&#x2020;</mo>
  </msup>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>i</mi>
        </mtd>
        <mtd>
          <mo>&#x2212;</mo>
          <mi>i</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mo>&#x2212;</mo>
          <mi>i</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mi>i</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>1</mn>
          <mo>&#x22C5;</mo>
          <mn>1</mn>
          <mo>+</mo>
          <mn>1</mn>
          <mo>&#x22C5;</mo>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mn>1</mn>
          <mo>&#x22C5;</mo>
          <mo stretchy="false">(</mo>
          <mo>&#x2212;</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
          <mo>+</mo>
          <mn>1</mn>
          <mo>&#x22C5;</mo>
          <mi>i</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>i</mi>
          <mo>&#x22C5;</mo>
          <mn>1</mn>
          <mo>+</mo>
          <mo stretchy="false">(</mo>
          <mo>&#x2212;</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
          <mo>&#x22C5;</mo>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mi>i</mi>
          <mo>&#x22C5;</mo>
          <mo stretchy="false">(</mo>
          <mo>&#x2212;</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
          <mo>+</mo>
          <mo stretchy="false">(</mo>
          <mo>&#x2212;</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
          <mo>&#x22C5;</mo>
          <mi>i</mi>
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
          <mn>1</mn>
        </mtd>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>

If the multiplication result 𝐴𝐴† is an identity matrix, which is indeed the case, and the product 𝐴†𝐴 is also an identity matrix (which you can verify in a similar manner), the matrix is unitary.


### Inner Product
---
The **inner product** is yet another important matrix operation that is only applied to vectors. Given two vectors 𝑉 and 𝑊 of the same size, their inner product ⟨𝑉,𝑊⟩ is defined as a product of matrices 𝑉† and 𝑊:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mi>V</mi>
  <mo>,</mo>
  <mi>W</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <msup>
    <mi>V</mi>
    <mo>&#x2020;</mo>
  </msup>
  <mi>W</mi>
</math>
Let's break this down so it's a bit easier to understand. A 1×𝑛 matrix (the adjoint of an 𝑛×1 vector) multiplied by an 𝑛×1 vector results in a 1×1 matrix (which is equivalent to a scalar). The result of an inner product is that scalar.

If you are familiar with the **dot product**, you will notice that it is equivalent to inner product for real-numbered vectors.

>We use our definition for these tutorials because it matches the notation used in quantum computing. You might encounter other sources which define the inner product a little differently: <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mo fence="false" stretchy="false">&#x27E8;</mo>
  <mi>V</mi>
  <mo>,</mo>
  <mi>W</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <msup>
    <mi>W</mi>
    <mo>&#x2020;</mo>
  </msup>
  <mi>V</mi>
  <mo>=</mo>
  <msup>
    <mi>V</mi>
    <mi>T</mi>
  </msup>
  <mover>
    <mi>W</mi>
    <mo accent="true">&#x2015;</mo>
  </mover>
</math>, in contrast to the 𝑉†𝑊 that we use. These definitions are almost equivalent, with some differences in the scalar multiplication by a complex number.

#### Vector Norm
---

An immediate application for the inner product is computing the **vector norm**. The norm of vector 𝑉 is defined as <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>V</mi>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mo>=</mo>
  <msqrt>
    <mo fence="false" stretchy="false">&#x27E8;</mo>
    <mi>V</mi>
    <mo>,</mo>
    <mi>V</mi>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </msqrt>
</math>. This condenses the vector down to a single non-negative real value. If the vector represents coordinates in space, the norm happens to be the length of the vector. A vector is called **normalized** if its norm is equal to 1.

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

The **outer product** of two vectors 𝑉 and 𝑊 is defined as 𝑉𝑊†. That is, the outer product of an 𝑛×1 vector and an 𝑚×1 vector is an 𝑛×𝑚 matrix. If we denote the outer product of 𝑉 and 𝑊 as 𝑋, then <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>X</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>,</mo>
      <mi>j</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <msub>
    <mi>V</mi>
    <mi>i</mi>
  </msub>
  <mo>&#x22C5;</mo>
  <mover>
    <msub>
      <mi>W</mi>
      <mi>j</mi>
    </msub>
    <mo accent="true">&#x2015;</mo>
  </mover>
</math>.

### Tensor Products
---

The **tensor product** is a different way of multiplying matrices. Rather than multiplying rows by columns, the tensor product multiplies the second matrix by every element of the first matrix.

Given 𝑛×𝑚 matrix 𝐴 and 𝑘×𝑙 matrix 𝐵, their tensor product 𝐴⊗𝐵 is an (𝑛⋅𝑘)×(𝑚⋅𝑙) matrix defined as follows:

![[Pasted image 20240625163207.png]]

Notice that the tensor product of two vectors is another vector: if 𝑉 is an 𝑛×1 vector, and 𝑊 is an 𝑚×1 vector, 𝑉⊗𝑊 is an (𝑛⋅𝑚)×1 vector.

The tensor product has the following properties:

- Distributivity over addition: <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mo stretchy="false">(</mo>
  <mi>A</mi>
  <mo>+</mo>
  <mi>B</mi>
  <mo stretchy="false">)</mo>
  <mo>&#x2297;</mo>
  <mi>C</mi>
  <mo>=</mo>
  <mi>A</mi>
  <mo>&#x2297;</mo>
  <mi>C</mi>
  <mo>+</mo>
  <mi>B</mi>
  <mo>&#x2297;</mo>
  <mi>C</mi>
</math> <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>A</mi>
  <mo>&#x2297;</mo>
  <mo stretchy="false">(</mo>
  <mi>B</mi>
  <mo>+</mo>
  <mi>C</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mi>A</mi>
  <mo>&#x2297;</mo>
  <mi>B</mi>
  <mo>+</mo>
  <mi>A</mi>
  <mo>&#x2297;</mo>
  <mi>C</mi>
</math>
- Associativity with scalar multiplication: <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>x</mi>
  <mo stretchy="false">(</mo>
  <mi>A</mi>
  <mo>&#x2297;</mo>
  <mi>B</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mi>A</mi>
  <mo stretchy="false">)</mo>
  <mo>&#x2297;</mo>
  <mi>B</mi>
  <mo>=</mo>
  <mi>A</mi>
  <mo>&#x2297;</mo>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mi>B</mi>
  <mo stretchy="false">)</mo>
</math>
- Mixed-product property (relation with matrix multiplication):<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mo stretchy="false">(</mo>
  <mi>A</mi>
  <mo>&#x2297;</mo>
  <mi>B</mi>
  <mo stretchy="false">)</mo>
  <mo stretchy="false">(</mo>
  <mi>C</mi>
  <mo>&#x2297;</mo>
  <mi>D</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mo stretchy="false">(</mo>
  <mi>A</mi>
  <mi>C</mi>
  <mo stretchy="false">)</mo>
  <mo>&#x2297;</mo>
  <mo stretchy="false">(</mo>
  <mi>B</mi>
  <mi>D</mi>
  <mo stretchy="false">)</mo>
</math>
