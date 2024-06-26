
---
> Most of these notes are provided by Microsoft [here](https://quantum.microsoft.com/en-us/experience/quantum-katas) 
> Another good source that details much found on here can be found on [this page](https://en.wikipedia.org/wiki/Complex_number)


Consider the expression 𝑒𝑖𝜃=cos⁡𝜃+𝑖sin⁡𝜃. Notice that if we map this number onto the complex plane, it will land on a **unit circle** around 0+0𝑖. This means that its modulus is always 1. You can also verify this algebraically: cos2⁡𝜃+sin2⁡𝜃=1.

Using this fact we can represent complex numbers using **polar coordinates**. In a polar coordinate system, a point is represented by two numbers: its direction from origin, represented by an angle from the 𝑥 axis, and how far away it is in that direction.

Another way to think about this is that we're taking a point that is 1 unit away (which is on the unit circle) in the specified direction, and multiplying it by the desired distance. And to get the point on the unit circle, we can use 𝑒𝑖𝜃.

A complex number of the format 𝑟⋅𝑒𝑖𝜃 will be represented by a point which is 𝑟 units away from the origin, in the direction specified by the angle 𝜃. Sometimes 𝜃 will be referred to as the number's **argument** or **phase**.

>In Q#, complex numbers in polar form are represented as user-defined type `ComplexPolar` from the `Microsoft.Quantum.Math` namespace.
>
>You can convert a complex number 𝑥=𝑟⋅𝑒𝑖𝜃 into a tuple of two `Double` numbers using unwrap operator and tuple deconstruction: `let (r, theta) = x!;`, or access its magnitude and phase using their names: `let (r, theta) = (x::Magnitude, x::Argument);`.
>
>You can construct a complex number from its magnitude and phase as follows: `let x = ComplexPolar(r, theta);`.

This method demonstrates how to convert regular complex numbers of format `x=a+bi` into a polar complex number:

---

### Convert Cartesian To Polar

We need to calculate the 𝑟 and 𝜃 values as seen in the complex plane. 𝑟 should be familiar to you already, since it is the modulus of a number (exercise 6):
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>r</mi>
  <mo>=</mo>
  <msqrt>
    <msup>
      <mi>a</mi>
      <mn>2</mn>
    </msup>
    <mo>+</mo>
    <msup>
      <mi>b</mi>
      <mn>2</mn>
    </msup>
  </msqrt>
</math>
  
 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>&#x3B8;</mi>
</math> can be calculated using trigonometry: since we know that the polar and the Cartesian forms of the number represent the same value, we can write
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>r</mi>
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>&#x3B8;</mi>
    </mrow>
  </msup>
  <mo>=</mo>
  <mi>a</mi>
  <mo>+</mo>
  <mi>b</mi>
  <mi>i</mi>
</math>
Euler's formula allows us to express the left part of the equation as
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>r</mi>
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>&#x3B8;</mi>
    </mrow>
  </msup>
  <mo>=</mo>
  <mi>r</mi>
  <mi>cos</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>&#x3B8;</mi>
  <mo>+</mo>
  <mi>i</mi>
  <mi>r</mi>
  <mi>sin</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>&#x3B8;</mi>
</math>
For two complex numbers to be equal, their real and imaginary parts have to be equal. This gives us the following system of equations:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">{</mo>
    <mtable columnalign="left left" columnspacing="1em" rowspacing=".2em">
      <mtr>
        <mtd>
          <mi>a</mi>
          <mo>=</mo>
          <mi>r</mi>
          <mi>cos</mi>
          <mo data-mjx-texclass="NONE">&#x2061;</mo>
          <mi>&#x3B8;</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>b</mi>
          <mo>=</mo>
          <mi>r</mi>
          <mi>sin</mi>
          <mo data-mjx-texclass="NONE">&#x2061;</mo>
          <mi>&#x3B8;</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE" fence="true" stretchy="true" symmetric="true"></mo>
  </mrow>
</math>
To calculate 𝜃, we can divide the second equation by the first one to get
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>tan</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>&#x3B8;</mi>
  <mo>=</mo>
  <mfrac>
    <mi>b</mi>
    <mi>a</mi>
  </mfrac>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>&#x3B8;</mi>
  <mo>=</mo>
  <mi>arctan</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">(</mo>
    <mfrac>
      <mi>b</mi>
      <mi>a</mi>
    </mfrac>
    <mo data-mjx-texclass="CLOSE">)</mo>
  </mrow>
</math>
This is the Q# method that can apply this function:

```C#
namespace Kata {
    open Microsoft.Quantum.Math;
    
    function ComplexToComplexPolar(x : Complex) : ComplexPolar {
        let (a, b) = x!;
        return ComplexPolar(Sqrt(a * a + b * b), ArcTan2(b, a));
    }
}

```

--- 

### Complex Polar to Complex Cartesian

Euler's formula allows us to express the complex polar number as

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>r</mi>
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>&#x3B8;</mi>
    </mrow>
  </msup>
  <mo>=</mo>
  <munder>
    <mrow data-mjx-texclass="OP">
      <munder>
        <mrow>
          <mi>r</mi>
          <mi>cos</mi>
          <mo data-mjx-texclass="NONE">&#x2061;</mo>
          <mi>&#x3B8;</mi>
        </mrow>
        <mo>&#x23DF;</mo>
      </munder>
    </mrow>
    <mrow>
      <mi>r</mi>
      <mi>e</mi>
      <mi>a</mi>
      <mi>l</mi>
    </mrow>
  </munder>
  <mo>+</mo>
  <mi>i</mi>
  <munder>
    <mrow data-mjx-texclass="OP">
      <munder>
        <mrow>
          <mi>r</mi>
          <mi>sin</mi>
          <mo data-mjx-texclass="NONE">&#x2061;</mo>
          <mi>&#x3B8;</mi>
        </mrow>
        <mo>&#x23DF;</mo>
      </munder>
    </mrow>
    <mrow>
      <mi>i</mi>
      <mi>m</mi>
      <mi>a</mi>
      <mi>g</mi>
      <mi>i</mi>
      <mi>n</mi>
      <mi>a</mi>
      <mi>r</mi>
      <mi>y</mi>
    </mrow>
  </munder>
</math>This is the Q# method that completes this :

```C#
namespace Kata {
    open Microsoft.Quantum.Math;
    
    function ComplexPolarToComplex(x : ComplexPolar) : Complex {
        let (r, theta) = x!;

        return Complex(r*Cos(theta), r*Sin(theta));
    }
}
```

---

### Multiplying Complex Polar Numbers

Multiplying two complex numbers in polar form can be done efficiently in the following way:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>z</mi>
  <mo>=</mo>
  <mi>x</mi>
  <mo>&#x22C5;</mo>
  <mi>y</mi>
  <mo>=</mo>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
  </msub>
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <msub>
        <mi>&#x3B8;</mi>
        <mn>1</mn>
      </msub>
      <mi>i</mi>
    </mrow>
  </msup>
  <mo>&#x22C5;</mo>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>2</mn>
    </mrow>
  </msub>
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <msub>
        <mi>&#x3B8;</mi>
        <mn>2</mn>
      </msub>
      <mi>i</mi>
    </mrow>
  </msup>
  <mo>=</mo>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
  </msub>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>2</mn>
    </mrow>
  </msub>
  <mo>&#x22C5;</mo>
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">(</mo>
      <msub>
        <mi>&#x3B8;</mi>
        <mn>1</mn>
      </msub>
      <mo>+</mo>
      <msub>
        <mi>&#x3B8;</mi>
        <mn>2</mn>
      </msub>
      <mo stretchy="false">)</mo>
      <mi>i</mi>
    </mrow>
  </msup>
</math>Here is the longer approach of converting the numbers to the Cartesian from and doing multiplication in it:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>x</mi>
  <mo>=</mo>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
  </msub>
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <msub>
        <mi>&#x3B8;</mi>
        <mn>1</mn>
      </msub>
    </mrow>
  </msup>
  <mo>=</mo>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>cos</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <msub>
    <mi>&#x3B8;</mi>
    <mn>1</mn>
  </msub>
  <mo>+</mo>
  <mi>i</mi>
  <mi>sin</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <msub>
    <mi>&#x3B8;</mi>
    <mn>1</mn>
  </msub>
  <mo stretchy="false">)</mo>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>y</mi>
  <mo>=</mo>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>2</mn>
    </mrow>
  </msub>
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <msub>
        <mi>&#x3B8;</mi>
        <mn>2</mn>
      </msub>
    </mrow>
  </msup>
  <mo>=</mo>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>2</mn>
    </mrow>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>cos</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <msub>
    <mi>&#x3B8;</mi>
    <mn>2</mn>
  </msub>
  <mo>+</mo>
  <mi>i</mi>
  <mi>sin</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <msub>
    <mi>&#x3B8;</mi>
    <mn>2</mn>
  </msub>
  <mo stretchy="false">)</mo>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>z</mi>
  <mo>=</mo>
  <mi>x</mi>
  <mo>&#x22C5;</mo>
  <mi>y</mi>
  <mo>=</mo>
  <msub>
    <mi>r</mi>
    <mn>1</mn>
  </msub>
  <msub>
    <mi>r</mi>
    <mn>2</mn>
  </msub>
  <mo>&#x22C5;</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">(</mo>
    <mi>cos</mi>
    <mo data-mjx-texclass="NONE">&#x2061;</mo>
    <msub>
      <mi>&#x3B8;</mi>
      <mn>1</mn>
    </msub>
    <mi>cos</mi>
    <mo data-mjx-texclass="NONE">&#x2061;</mo>
    <msub>
      <mi>&#x3B8;</mi>
      <mn>2</mn>
    </msub>
    <mo>&#x2212;</mo>
    <mi>sin</mi>
    <mo data-mjx-texclass="NONE">&#x2061;</mo>
    <msub>
      <mi>&#x3B8;</mi>
      <mn>1</mn>
    </msub>
    <mi>sin</mi>
    <mo data-mjx-texclass="NONE">&#x2061;</mo>
    <msub>
      <mi>&#x3B8;</mi>
      <mn>2</mn>
    </msub>
    <mo>+</mo>
    <mi>i</mi>
    <mo stretchy="false">(</mo>
    <mi>sin</mi>
    <mo data-mjx-texclass="NONE">&#x2061;</mo>
    <msub>
      <mi>&#x3B8;</mi>
      <mn>1</mn>
    </msub>
    <mi>cos</mi>
    <mo data-mjx-texclass="NONE">&#x2061;</mo>
    <msub>
      <mi>&#x3B8;</mi>
      <mn>2</mn>
    </msub>
    <mo>+</mo>
    <mi>sin</mi>
    <mo data-mjx-texclass="NONE">&#x2061;</mo>
    <msub>
      <mi>&#x3B8;</mi>
      <mn>2</mn>
    </msub>
    <mi>cos</mi>
    <mo data-mjx-texclass="NONE">&#x2061;</mo>
    <msub>
      <mi>&#x3B8;</mi>
      <mn>1</mn>
    </msub>
    <mo stretchy="false">)</mo>
    <mo data-mjx-texclass="CLOSE">)</mo>
  </mrow>
</math>
We can simplify this using the following trigonometric identities:
- <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>cos</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>a</mi>
  <mi>cos</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>b</mi>
  <mo>&#x2213;</mo>
  <mi>sin</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>a</mi>
  <mi>sin</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>b</mi>
  <mo>=</mo>
  <mi>cos</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mo stretchy="false">(</mo>
  <mi>a</mi>
  <mo>&#xB1;</mo>
  <mi>b</mi>
  <mo stretchy="false">)</mo>
</math>
- <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>sin</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>a</mi>
  <mi>cos</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>b</mi>
  <mo>&#xB1;</mo>
  <mi>sin</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>b</mi>
  <mi>cos</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>a</mi>
  <mo>=</mo>
  <mi>sin</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mo stretchy="false">(</mo>
  <mi>a</mi>
  <mo>&#xB1;</mo>
  <mi>b</mi>
  <mo stretchy="false">)</mo>
</math>
Finally, this solution gives the same answer as the short solution above:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
  </msub>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>2</mn>
    </mrow>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>cos</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>&#x3B8;</mi>
    <mn>1</mn>
  </msub>
  <mo>+</mo>
  <msub>
    <mi>&#x3B8;</mi>
    <mn>2</mn>
  </msub>
  <mo stretchy="false">)</mo>
  <mo>+</mo>
  <mi>i</mi>
  <mi>sin</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>&#x3B8;</mi>
    <mn>1</mn>
  </msub>
  <mo>+</mo>
  <msub>
    <mi>&#x3B8;</mi>
    <mn>2</mn>
  </msub>
  <mo stretchy="false">)</mo>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
  </msub>
  <msub>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>2</mn>
    </mrow>
  </msub>
  <mo>&#x22C5;</mo>
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">(</mo>
      <msub>
        <mi>&#x3B8;</mi>
        <mn>1</mn>
      </msub>
      <mo>+</mo>
      <msub>
        <mi>&#x3B8;</mi>
        <mn>2</mn>
      </msub>
      <mo stretchy="false">)</mo>
      <mi>i</mi>
    </mrow>
  </msup>
</math>
This is the Q# method that implements this :

```C#
namespace Kata {
    open Microsoft.Quantum.Math;
    
    function ComplexPolarMult(x : ComplexPolar, y: ComplexPolar) : ComplexPolar {
        let (r1, t1) = x!;
        let (r2, t2) = y!;

        mutable t3 = t1 + t2;
        if t3 > PI() {
            set t3 -= 2.0 * PI();
        }
        if t3 < -PI() {
            set t3 += 2.0 * PI();
        }
        return ComplexPolar(r1*r2,t3);
    }
}

```