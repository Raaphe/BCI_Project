
---

> Most of these notes are provided by Microsoft [here](https://quantum.microsoft.com/en-us/experience/quantum-katas) 
> Another good source that details much found on here can be found on [this page](https://en.wikipedia.org/wiki/Complex_number)

Adding imaginary numbers to each other is quite simple, but what happens when we add a real number to an imaginary number? The result of that addition will be partly real and partly imaginary, otherwise known as a **complex number**. A complex number is simply the real part and the imaginary part being treated as a single number. Complex numbers are generally written as the sum of their two parts 𝑎 and 𝑏𝑖, where both 𝑎 and 𝑏 are real numbers:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>a</mi>
  <mo>+</mo>
  <mi>b</mi>
  <mi>i</mi>
</math>
For example, 3+4𝑖 or −5−7𝑖 are valid complex numbers. Note that purely real or purely imaginary numbers can also be written as complex numbers: 2 is 2+0𝑖, and −3𝑖 is 0−3𝑖.

When performing operations on complex numbers, it is often helpful to treat them as polynomials in terms of 𝑖. Let's see how to do the main arithmetic operations on complex numbers.

>In Q#, complex numbers are represented as user-defined type `Complex` from the `Microsoft.Quantum.Math` namespace. 
>
>You can convert a complex number 𝑥=𝑎+𝑏𝑖 into a tuple of two `Double` numbers using unwrap operator and tuple deconstruction: `let (a, b) = x!;`, or access its real and imaginary parts using their names: `let (a, b) = (x::Real, x::Imag);`. 
> 
   You can construct a complex number from its real and imaginary parts as follows: `let x = Complex(a, b);`.


---
### Addition

Here is a function that shows how to add complex numbers together using this formula : 

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>x</mi>
  <mo>+</mo>
  <mi>y</mi>
  <mo>=</mo>
  <mo stretchy="false">(</mo>
  <mi>a</mi>
  <mo>+</mo>
  <mi>b</mi>
  <mi>i</mi>
  <mo stretchy="false">)</mo>
  <mo>+</mo>
  <mo stretchy="false">(</mo>
  <mi>c</mi>
  <mo>+</mo>
  <mi>d</mi>
  <mi>i</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <munder>
    <mrow data-mjx-texclass="OP">
      <munder>
        <mrow>
          <mo stretchy="false">(</mo>
          <mi>a</mi>
          <mo>+</mo>
          <mi>c</mi>
          <mo stretchy="false">)</mo>
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
  <munder>
    <mrow data-mjx-texclass="OP">
      <munder>
        <mrow>
          <mo stretchy="false">(</mo>
          <mi>b</mi>
          <mo>+</mo>
          <mi>d</mi>
          <mo stretchy="false">)</mo>
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
  <mi>i</mi>
</math>


```c#
namespace Kata {    
    open Microsoft.Quantum.Math;
     
    function ComplexAdd(x : Complex, y: Complex) : Complex {        
        let (a, b) = x!;
        let (c, d) = y!;
        return Complex(a + c, b + d);
    }
}
```


---
### Multiplication

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>x</mi>
  <mo>&#x22C5;</mo>
  <mi>y</mi>
  <mo>=</mo>
  <mo stretchy="false">(</mo>
  <mi>a</mi>
  <mo>+</mo>
  <mi>b</mi>
  <mi>i</mi>
  <mo stretchy="false">)</mo>
  <mo stretchy="false">(</mo>
  <mi>c</mi>
  <mo>+</mo>
  <mi>d</mi>
  <mi>i</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mi>a</mi>
  <mo>&#x22C5;</mo>
  <mi>c</mi>
  <mo>+</mo>
  <mi>a</mi>
  <mo>&#x22C5;</mo>
  <mi>d</mi>
  <mi>i</mi>
  <mo>+</mo>
  <mi>c</mi>
  <mo>&#x22C5;</mo>
  <mi>b</mi>
  <mi>i</mi>
  <mo>+</mo>
  <mi>b</mi>
  <mi>i</mi>
  <mo>&#x22C5;</mo>
  <mi>d</mi>
  <mi>i</mi>
  <mo>=</mo>
  <munder>
    <mrow data-mjx-texclass="OP">
      <munder>
        <mrow>
          <mi>a</mi>
          <mo>&#x22C5;</mo>
          <mi>c</mi>
          <mo>&#x2212;</mo>
          <mi>b</mi>
          <mo>&#x22C5;</mo>
          <mi>d</mi>
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
  <munder>
    <mrow data-mjx-texclass="OP">
      <munder>
        <mrow>
          <mo stretchy="false">(</mo>
          <mi>a</mi>
          <mo>&#x22C5;</mo>
          <mi>d</mi>
          <mo>+</mo>
          <mi>c</mi>
          <mo>&#x22C5;</mo>
          <mi>b</mi>
          <mo stretchy="false">)</mo>
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
  <mi>i</mi>
</math>

```C#
namespace Kata {    
    open Microsoft.Quantum.Math;
    
    function ComplexMult(x : Complex, y: Complex) : Complex {
        let (a, b) = x!;
        let (c, d) = y!;
        return Complex(a * c - b * d, a * d + b * c);
    }
}
```

---

## Complex Conjugates

Before we discuss any other operations on complex numbers, we have to cover the **complex conjugate**. The conjugate is a simple operation: given a complex number 𝑥=𝑎+𝑏𝑖, its complex conjugate is 𝑥―=𝑎−𝑏𝑖.

The conjugate allows us to do some interesting things. The first and probably most important is multiplying a complex number by its conjugate:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>x</mi>
  <mo>&#x22C5;</mo>
  <mover>
    <mi>x</mi>
    <mo accent="true">&#x2015;</mo>
  </mover>
  <mo>=</mo>
  <mo stretchy="false">(</mo>
  <mi>a</mi>
  <mo>+</mo>
  <mi>b</mi>
  <mi>i</mi>
  <mo stretchy="false">)</mo>
  <mo stretchy="false">(</mo>
  <mi>a</mi>
  <mo>&#x2212;</mo>
  <mi>b</mi>
  <mi>i</mi>
  <mo stretchy="false">)</mo>
</math>

Notice that the second expression is a difference of squares:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo stretchy="false">(</mo>
  <mi>a</mi>
  <mo>+</mo>
  <mi>b</mi>
  <mi>i</mi>
  <mo stretchy="false">)</mo>
  <mo stretchy="false">(</mo>
  <mi>a</mi>
  <mo>&#x2212;</mo>
  <mi>b</mi>
  <mi>i</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <msup>
    <mi>a</mi>
    <mn>2</mn>
  </msup>
  <mo>&#x2212;</mo>
  <mo stretchy="false">(</mo>
  <mi>b</mi>
  <mi>i</mi>
  <msup>
    <mo stretchy="false">)</mo>
    <mn>2</mn>
  </msup>
  <mo>=</mo>
  <msup>
    <mi>a</mi>
    <mn>2</mn>
  </msup>
  <mo>&#x2212;</mo>
  <msup>
    <mi>b</mi>
    <mn>2</mn>
  </msup>
  <msup>
    <mi>i</mi>
    <mn>2</mn>
  </msup>
  <mo>=</mo>
  <msup>
    <mi>a</mi>
    <mn>2</mn>
  </msup>
  <mo>+</mo>
  <msup>
    <mi>b</mi>
    <mn>2</mn>
  </msup>
</math>

This means that a complex number multiplied by its conjugate always produces a non-negative real number.

Another property of the conjugate is that it distributes over both complex addition and complex multiplication:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mover>
    <mrow>
      <mi>x</mi>
      <mo>+</mo>
      <mi>y</mi>
    </mrow>
    <mo accent="true">&#x2015;</mo>
  </mover>
  <mo>=</mo>
  <mover>
    <mi>x</mi>
    <mo accent="true">&#x2015;</mo>
  </mover>
  <mo>+</mo>
  <mover>
    <mi>y</mi>
    <mo accent="true">&#x2015;</mo>
  </mover>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mover>
    <mrow>
      <mi>x</mi>
      <mo>&#x22C5;</mo>
      <mi>y</mi>
    </mrow>
    <mo accent="true">&#x2015;</mo>
  </mover>
  <mo>=</mo>
  <mover>
    <mi>x</mi>
    <mo accent="true">&#x2015;</mo>
  </mover>
  <mo>&#x22C5;</mo>
  <mover>
    <mi>y</mi>
    <mo accent="true">&#x2015;</mo>
  </mover>
</math>
This is the Q# method to find a complex number's conjugate :

```C#
namespace Kata {    
    open Microsoft.Quantum.Math;

    function ComplexConjugate(x : Complex) : Complex {        
        let (a,b) = x!;
        return Complex(a, b * -1.);
    }
}
```

---

## Complex Division

The next use for the conjugate is complex division. Let's take two complex numbers: 𝑥=𝑎+𝑏𝑖 and 𝑦=𝑐+𝑑𝑖≠0 (not even complex numbers let you divide by 0). What does 𝑥𝑦 mean?

Let's expand 𝑥 and 𝑦 into their component forms:The next use for the conjugate is complex division. Let's take two complex numbers: 𝑥=𝑎+𝑏𝑖 and 𝑦=𝑐+𝑑𝑖≠0 (not even complex numbers let you divide by 0). What does 𝑥𝑦 mean?

Let's expand 𝑥 and 𝑦 into their component forms:\
\
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mfrac>
    <mi>x</mi>
    <mi>y</mi>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>a</mi>
      <mo>+</mo>
      <mi>b</mi>
      <mi>i</mi>
    </mrow>
    <mrow>
      <mi>c</mi>
      <mo>+</mo>
      <mi>d</mi>
      <mi>i</mi>
    </mrow>
  </mfrac>
</math>

Unfortunately, it isn't very clear what it means to divide by a complex number. We need some way to move either all real parts or all imaginary parts into the numerator. And thanks to the conjugate, we can do just that. Using the fact that any number (except 0) divided by itself equals 1, and any number multiplied by 1 equals itself, we get:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mfrac>
    <mi>x</mi>
    <mi>y</mi>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mi>x</mi>
    <mi>y</mi>
  </mfrac>
  <mo>&#x22C5;</mo>
  <mn>1</mn>
  <mo>=</mo>
  <mfrac>
    <mi>x</mi>
    <mi>y</mi>
  </mfrac>
  <mo>&#x22C5;</mo>
  <mfrac>
    <mover>
      <mi>y</mi>
      <mo accent="true">&#x2015;</mo>
    </mover>
    <mover>
      <mi>y</mi>
      <mo accent="true">&#x2015;</mo>
    </mover>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>x</mi>
      <mover>
        <mi>y</mi>
        <mo accent="true">&#x2015;</mo>
      </mover>
    </mrow>
    <mrow>
      <mi>y</mi>
      <mover>
        <mi>y</mi>
        <mo accent="true">&#x2015;</mo>
      </mover>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mo stretchy="false">(</mo>
      <mi>a</mi>
      <mo>+</mo>
      <mi>b</mi>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
      <mo stretchy="false">(</mo>
      <mi>c</mi>
      <mo>&#x2212;</mo>
      <mi>d</mi>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
    <mrow>
      <mo stretchy="false">(</mo>
      <mi>c</mi>
      <mo>+</mo>
      <mi>d</mi>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
      <mo stretchy="false">(</mo>
      <mi>c</mi>
      <mo>&#x2212;</mo>
      <mi>d</mi>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mo stretchy="false">(</mo>
      <mi>a</mi>
      <mo>+</mo>
      <mi>b</mi>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
      <mo stretchy="false">(</mo>
      <mi>c</mi>
      <mo>&#x2212;</mo>
      <mi>d</mi>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
    <mrow>
      <msup>
        <mi>c</mi>
        <mn>2</mn>
      </msup>
      <mo>+</mo>
      <msup>
        <mi>d</mi>
        <mn>2</mn>
      </msup>
    </mrow>
  </mfrac>
</math>
By doing this, we re-wrote our division problem to have a complex multiplication expression in the numerator, and a real number in the denominator. We already know how to multiply complex numbers, and dividing a complex number by a real number is as simple as dividing both parts of the complex number separately:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mfrac>
    <mrow>
      <mi>a</mi>
      <mo>+</mo>
      <mi>b</mi>
      <mi>i</mi>
    </mrow>
    <mi>r</mi>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mi>a</mi>
    <mi>r</mi>
  </mfrac>
  <mo>+</mo>
  <mfrac>
    <mi>b</mi>
    <mi>r</mi>
  </mfrac>
  <mi>i</mi>
</math>
This is the Q# method that divides two complex numbers :

```C#
namespace Kata {    
    open Microsoft.Quantum.Math;
    
    function ComplexDiv(x : Complex, y: Complex) : Complex {
        let (a, b) = x!;
        let (c, d) = y!;
        let denominator = c * c + d * d;
        let real = (a * c + b * d) / denominator;
        let imag = (- a * d + b * c) / denominator;
        return Complex(real, imag);
    }
}
```


--- 

## Geometric Perspective: The Complex Plane

![[Pasted image 20240622112914.png]]

You may recall that real numbers can be represented geometrically using the number line - a line on which each point represents a real number. We can extend this representation to include imaginary and complex numbers, which gives rise to an entirely different number line: the imaginary number line, which is orthogonal to the real number line and only intersects with it at 0.

A complex number has two components - a real component and an imaginary component. As you no doubt noticed from the exercises, these can be represented by two real numbers - the real component, and the real coefficient of the imaginary component. This allows us to map complex numbers onto a two-dimensional plane - the **complex plane**. The most common mapping is the obvious one: 𝑎+𝑏𝑖 can be represented by the point (𝑎,𝑏) in the **Cartesian coordinate system**.

This mapping allows us to apply complex arithmetic to geometry, and, more importantly, apply geometric concepts to complex numbers. Many properties of complex numbers become easier to understand when viewed through a geometric lens.

---

## Modulus

One such property is the **modulus operator**. This operator generalizes the **absolute value** operator on real numbers to the complex plane. Just like the absolute value of a number is its distance from 0, the modulus of a complex number is its distance from 0+0𝑖. Using the distance formula, if 𝑥=𝑎+𝑏𝑖, then:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>x</mi>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
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
There is also a slightly different, but algebraically equivalent definition:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>x</mi>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mo>=</mo>
  <msqrt>
    <mi>x</mi>
    <mo>&#x22C5;</mo>
    <mover>
      <mi>x</mi>
      <mo accent="true">&#x2015;</mo>
    </mover>
  </msqrt>
</math>
Like the conjugate, the modulus distributes over multiplication.****
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>x</mi>
  <mo>&#x22C5;</mo>
  <mi>y</mi>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mo>=</mo>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>x</mi>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mo>&#x22C5;</mo>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>y</mi>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
</math>
Unlike the conjugate, however, the modulus doesn't distribute over addition. Instead, the interaction of the two comes from the triangle inequality:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>x</mi>
  <mo>+</mo>
  <mi>y</mi>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mo>&#x2264;</mo>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>x</mi>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mo>+</mo>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>y</mi>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
</math>
The following method finds the modulus of a complex number :

```C#
namespace Kata {
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Intrinsic;

    function ComplexModulus(x : Complex) : Double {
        return Sqrt((x::Real * x::Real) + (x::Imag * x::Imag));
    }
}
```

---

## Power of i

![[Pasted image 20240622153503.png]]

The next complex operation we're going to need is exponentiation. Raising an imaginary number to an integer power is a fairly simple task, but raising a number to an imaginary power, or raising an imaginary (or complex) number to a real power isn't quite as simple.

Let's start with raising real numbers to imaginary powers. Specifically, let's start with a rather special real number - Euler's constant, 𝑒:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>&#x3B8;</mi>
    </mrow>
  </msup>
  <mo>=</mo>
  <mi>cos</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>&#x3B8;</mi>
  <mo>+</mo>
  <mi>i</mi>
  <mi>sin</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>&#x3B8;</mi>
</math>

Here and later in this tutorial 𝜃 is measured in radians.

> Explaining why that happens is somewhat beyond the scope of this tutorial, as it requires some calculus, so we won't do that here. If you are curious, you can see [this video](https://youtu.be/v0YEaeIClKY) for a beautiful intuitive explanation, or [this Wikipedia article](https://en.wikipedia.org/wiki/Complex_number) for a more mathematically rigorous proof.

Here are some examples of this formula in action:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>&#x3C0;</mi>
      <mrow data-mjx-texclass="ORD">
        <mo>/</mo>
      </mrow>
      <mn>4</mn>
    </mrow>
  </msup>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <msqrt>
      <mn>2</mn>
    </msqrt>
  </mfrac>
  <mo>+</mo>
  <mfrac>
    <mi>i</mi>
    <msqrt>
      <mn>2</mn>
    </msqrt>
  </mfrac>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>&#x3C0;</mi>
      <mrow data-mjx-texclass="ORD">
        <mo>/</mo>
      </mrow>
      <mn>2</mn>
    </mrow>
  </msup>
  <mo>=</mo>
  <mi>i</mi>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>&#x3C0;</mi>
    </mrow>
  </msup>
  <mo>=</mo>
  <mo>&#x2212;</mo>
  <mn>1</mn>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>2</mn>
      <mi>i</mi>
      <mi>&#x3C0;</mi>
    </mrow>
  </msup>
  <mo>=</mo>
  <mn>1</mn>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>&#x3C0;</mi>
    </mrow>
  </msup>
  <mo>+</mo>
  <mn>1</mn>
  <mo>=</mo>
  <mn>0</mn>
</math>

> One interesting consequence of this is Euler's identity:
> <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>&#x3C0;</mi>
    </mrow>
  </msup>
  <mo>+</mo>
  <mn>1</mn>
  <mo>=</mo>
  <mn>0</mn>
</math>
   While this doesn't have any notable uses, it is still an interesting identity to consider, as it combines five fundamental constants of algebra into one expression.


We can also calculate complex powers of 𝑒 as follows:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>a</mi>
      <mo>+</mo>
      <mi>b</mi>
      <mi>i</mi>
    </mrow>
  </msup>
  <mo>=</mo>
  <msup>
    <mi>e</mi>
    <mi>a</mi>
  </msup>
  <mo>&#x22C5;</mo>
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>b</mi>
      <mi>i</mi>
    </mrow>
  </msup>
</math>
Finally, using logarithms to express the base of the exponent as <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>r</mi>
  <mo>=</mo>
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>ln</mi>
      <mo data-mjx-texclass="NONE">&#x2061;</mo>
      <mi>r</mi>
    </mrow>
  </msup>
</math>, we can use this to find complex powers of any positive real number.

--- 

##### More in depth explanation on how to power euler's constant as a base

This formula is derived from the expression for a complex exponential ea+bie^{a + bi}ea+bi and the application of Euler's formula:

1.**Start with the Original Expression:**

<math xmlns="http://www.w3.org/1998/Math/MathML">
	  <!-- Start with the Original Expression: -->
	  <mrow>
		<msup>
		  <mi>e</mi>
		  <mrow>
			<mi>a</mi>
			<mo>+</mo>
			<mi>b</mi>
			<mi>i</mi>
		  </mrow>
		</msup>
	  </mrow>
  </math>

2 **Separate the Exponents Using Properties:** The exponential function can be split into a product of two simpler exponentials:

<math xmlns="http://www.w3.org/1998/Math/MathML">
	  <mrow>
	    <msup>
	      <mi>e</mi>
	      <mrow>
	        <mi>a</mi>
	        <mo>+</mo>
	        <mi>b</mi>
	        <mi>i</mi>
	      </mrow>
	    </msup>
	    <mo>=</mo>
	    <msup>
	      <mi>e</mi>
	      <mi>a</mi>
	    </msup>
	    <mo>&#x22C5;</mo>
	    <msup>
	      <mi>e</mi>
	      <mi>bi</mi>
	    </msup>
	  </mrow>
  </math>

3.**Apply Euler's Formula:** According to Euler's formula, ebie^{bi}ebi can be expressed as:

<math xmlns="http://www.w3.org/1998/Math/MathML">
	  <!-- Apply Euler's Formula: -->
	  <mrow>
	    <msup>
	      <mi>e</mi>
	      <mi>bi</mi>
	    </msup>
	    <mo>=</mo>
	    <mi>cos</mi>
	    <mo>&#x2061;</mo>
	    <mo>(</mo>
	    <mi>b</mi>
	    <mo>)</mo>
	    <mo>+</mo>
	    <mi>i</mi>
	    <mi>sin</mi>
	    <mo>&#x2061;</mo>
	    <mo>(</mo>
	    <mi>b</mi>
	    <mo>)</mo>
	  </mrow>
</math>

4.**Combine the Expressions:** Substitute Euler's formula into the separated exponential:

<math xmlns="http://www.w3.org/1998/Math/MathML">
  
  <!-- Combine the Expressions: -->
  <mrow>
    <msup>
      <mi>e</mi>
      <mrow>
        <mi>a</mi>
        <mo>+</mo>
        <mi>b</mi>
        <mi>i</mi>
      </mrow>
    </msup>
    <mo>=</mo>
    <msup>
      <mi>e</mi>
      <mi>a</mi>
    </msup>
    <mo>&#x22C5;</mo>
    <mo>(</mo>
    <mi>cos</mi>
    <mo>&#x2061;</mo>
    <mo>(</mo>
    <mi>b</mi>
    <mo>)</mo>
    <mo>+</mo>
    <mi>i</mi>
    <mi>sin</mi>
    <mo>&#x2061;</mo>
    <mo>(</mo>
    <mi>b</mi>
    <mo>)</mo>
    <mo>)</mo>
  </mrow>
</math>

This step results in this:

<math xmlns="http://www.w3.org/1998/Math/MathML">
	  <mrow>
	    <msup>
	      <mi>e</mi>
	      <mi>a</mi>
	    </msup>
	    <mo>(</mo>
	    <mi>cos</mi>
	    <mo>&#x2061;</mo>
	    <mo>(</mo>
	    <mi>b</mi>
	    <mo>)</mo>
	    <mo>+</mo>
	    <mi>i</mi>
	    <mi>sin</mi>
	    <mo>&#x2061;</mo>
	    <mo>(</mo>
	    <mi>b</mi>
	    <mo>)</mo>
	    <mo>)</mo>
	</mrow>
</math>


5.**Distribute <math xmlns="http://www.w3.org/1998/Math/MathML">

	  <!-- Distribute e^a: -->
	  <mrow>
	    <msup>
	      <mi>e</mi>
	      <mi>a</mi>
	    </msup>
	   </mrow>
</math>: When you distribute <math xmlns="http://www.w3.org/1998/Math/MathML">

	  <!-- Distribute e^a: -->
	  <mrow>
	    <msup>
	      <mi>e</mi>
	      <mi>a</mi>
	    </msup>
	   </mrow>
</math> across the terms inside the parentheses, you get: 

<math xmlns="http://www.w3.org/1998/Math/MathML">

  <!-- Distribute e^a: -->
  <mrow>
    <msup>
      <mi>e</mi>
      <mi>a</mi>
    </msup>
    <mo>(</mo>
    <mi>cos</mi>
    <mo>&#x2061;</mo>
    <mo>(</mo>
    <mi>b</mi>
    <mo>)</mo>
    <mo>+</mo>
    <mi>i</mi>
    <mi>sin</mi>
    <mo>&#x2061;</mo>
    <mo>(</mo>
    <mi>b</mi>
    <mo>)</mo>
    <mo>)</mo>
    <mo>=</mo>
    <msup>
      <mi>e</mi>
      <mi>a</mi>
    </msup>
    <mi>cos</mi>
    <mo>&#x2061;</mo>
    <mo>(</mo>
    <mi>b</mi>
    <mo>)</mo>
    <mo>+</mo>
    <msup>
      <mi>e</mi>
      <mi>a</mi>
    </msup>
    <mi>i</mi>
    <mi>sin</mi>
    <mo>&#x2061;</mo>
    <mo>(</mo>
    <mi>b</mi>
    <mo>)</mo>
  </mrow>
</math>

This is the Q# method for this :


```C#
namespace Kata {
    open Microsoft.Quantum.Math;
    
    function ComplexExponent(x : Complex) : Complex {
        let (a,b) = x!;
        return Complex(E()^a*(Cos(b)), E()^a*(Sin(b)));
    }
}
```

#### Complex Power of any Real Number

Rewrite the expression 𝑟𝑥 to use Euler's constant, which will allow us to use an approach similar to the solution to the previous exercise.

First, rewrite 𝑟𝑥 into a product of two powers:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>a</mi>
      <mo>+</mo>
      <mi>b</mi>
      <mi>i</mi>
    </mrow>
  </msup>
  <mo>=</mo>
  <msup>
    <mi>r</mi>
    <mi>a</mi>
  </msup>
  <mo>&#x22C5;</mo>
  <msup>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>b</mi>
      <mi>i</mi>
    </mrow>
  </msup>
</math>
Given that 𝑟=𝑒ln⁡𝑟 (ln is the natural logarithm), we can rewrite the second part of the product as follows:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>r</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>b</mi>
      <mi>i</mi>
    </mrow>
  </msup>
  <mo>=</mo>
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>b</mi>
      <mi>i</mi>
      <mi>ln</mi>
      <mo data-mjx-texclass="NONE">&#x2061;</mo>
      <mi>r</mi>
    </mrow>
  </msup>
</math>
Now, given 𝑒𝑖𝜃=cos⁡𝜃+𝑖sin⁡𝜃, we can rewrite it further as follows:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>b</mi>
      <mi>i</mi>
      <mi>ln</mi>
      <mo data-mjx-texclass="NONE">&#x2061;</mo>
      <mi>r</mi>
    </mrow>
  </msup>
  <mo>=</mo>
  <mi>cos</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mo stretchy="false">(</mo>
  <mi>b</mi>
  <mo>&#x22C5;</mo>
  <mi>ln</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>r</mi>
  <mo stretchy="false">)</mo>
  <mo>+</mo>
  <mi>i</mi>
  <mi>sin</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mo stretchy="false">(</mo>
  <mi>b</mi>
  <mo>&#x22C5;</mo>
  <mi>ln</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mi>r</mi>
  <mo stretchy="false">)</mo>
</math>
When substituting this into the original expression, we get:


<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <munder>
    <mrow data-mjx-texclass="OP">
      <munder>
        <mrow>
          <msup>
            <mi>r</mi>
            <mi>a</mi>
          </msup>
          <mi>cos</mi>
          <mo data-mjx-texclass="NONE">&#x2061;</mo>
          <mo stretchy="false">(</mo>
          <mi>b</mi>
          <mo>&#x22C5;</mo>
          <mi>ln</mi>
          <mo data-mjx-texclass="NONE">&#x2061;</mo>
          <mi>r</mi>
          <mo stretchy="false">)</mo>
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
  <munder>
    <mrow data-mjx-texclass="OP">
      <munder>
        <mrow>
          <msup>
            <mi>r</mi>
            <mi>a</mi>
          </msup>
          <mi>sin</mi>
          <mo data-mjx-texclass="NONE">&#x2061;</mo>
          <mo stretchy="false">(</mo>
          <mi>b</mi>
          <mo>&#x22C5;</mo>
          <mi>ln</mi>
          <mo data-mjx-texclass="NONE">&#x2061;</mo>
          <mi>r</mi>
          <mo stretchy="false">)</mo>
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
  <mi>i</mi>
</math>

This is the Q# method that implements this function :

```C#
namespace Kata { 
    open Microsoft.Quantum.Math;

    function ComplexExpReal(r : Double, x : Complex) : Complex {
        if AbsD(r) < 1e-9 {
            return Complex(0.0, 0.0);
        }
        
        let (a, b) = x!;        
        let ra = r ^ a;
        let lnr = Log(r);
        return Complex(ra * Cos(b * lnr), ra * Sin(b * lnr));
    }
}

```