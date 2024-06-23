
---

> Most of these notes are provided by Microsoft [here](https://quantum.microsoft.com/en-us/experience/quantum-katas)

For some purposes, real numbers aren't enough. Probably the most famous example is the equation:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>x</mi>
    <mrow data-mjx-texclass="ORD">
      <mn>2</mn>
    </mrow>
  </msup>
  <mo>=</mo>
  <mo>&#x2212;</mo>
  <mn>1</mn>
</math>

This equation has no solution among real numbers. If, however, we abandon that constraint, we can do something interesting - we can define our own number. Let's say there exists some number that solves that equation. Let's call that number 𝑖.
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>i</mi>
    <mn>2</mn>
  </msup>
  <mo>=</mo>
  <mo>&#x2212;</mo>
  <mn>1</mn>
</math>

As we said before, 𝑖 can't be a real number. In that case, we'll call it an **imaginary unit**. However, there is no reason for us to define it as acting any different from any other number, other than the fact that 𝑖2=−1:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>i</mi>
  <mo>+</mo>
  <mi>i</mi>
  <mo>=</mo>
  <mn>2</mn>
  <mi>i</mi>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>i</mi>
  <mo>&#x2212;</mo>
  <mi>i</mi>
  <mo>=</mo>
  <mn>0</mn>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo>&#x2212;</mo>
  <mn>1</mn>
  <mo>&#x22C5;</mo>
  <mi>i</mi>
  <mo>=</mo>
  <mo>&#x2212;</mo>
  <mi>i</mi>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo stretchy="false">(</mo>
  <mo>&#x2212;</mo>
  <mi>i</mi>
  <msup>
    <mo stretchy="false">)</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>2</mn>
    </mrow>
  </msup>
  <mo>=</mo>
  <mo>&#x2212;</mo>
  <mn>1</mn>
</math>
We'll call the number 𝑖 and its real multiples (numbers obtained by multiplying 𝑖 by a real number) **imaginary numbers**.

> A good video introduction to imaginary numbers can be found [here](https://www.youtube.com/watch?v=SP-YJe7Vldo&feature=youtu.be).

---

Here is a method that inputs a real whole number that will power an imaginary number. The method returns the value of the equation. 

```C#
namespace Kata {
    function PowersOfI(n : Int) : Int {
        if n % 4 == 0 {
            return 1;
        } else {
            return -1;
        }
    }
}
```

---
