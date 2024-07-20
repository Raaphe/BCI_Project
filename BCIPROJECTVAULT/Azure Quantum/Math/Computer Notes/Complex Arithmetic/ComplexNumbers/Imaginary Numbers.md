
---

> Most of these notes are provided by Microsoft [here](https://quantum.microsoft.com/en-us/experience/quantum-katas)

For some purposes, real numbers aren't enough. Probably the most famous example is the equation:
$$x^{2} = -1$$

This equation has no solution among real numbers. If, however, we abandon that constraint, we can do something interesting - we can define our own number. Let's say there exists some number that solves that equation. Let's call that number 𝑖.
$$i^2 = -1$$
As we said before, 𝑖 can't be a real number. In that case, we'll call it an **imaginary unit**. However, there is no reason for us to define it as acting any different from any other number, other than the fact that $i^2 = -1$:

$$i+i=2i$$
$$i-i=0$$
$$-1 \cdot i=-i$$
$$(-i)^{2} = -1$$


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
