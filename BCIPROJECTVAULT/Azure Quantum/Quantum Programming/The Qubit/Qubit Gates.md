
---

> Most of these notes are provided by Microsoft [here](https://quantum.microsoft.com/en-us/experience/quantum-katas) 
> Another good source that details much found on here can be found on [this page](https://en.wikipedia.org/wiki/Determinant)


## Matrix Representation
---

Quantum gates are represented as $2^N \times 2^N$ unitary matrices, where 𝑁 is the number of qubits the gate operates on. As a quick reminder, a unitary matrix is a square matrix whose inverse is its adjoint, thus $U^* U = UU^* = UU^{-1} = \mathbb{I}$. Single-qubit gates are represented by 2×2 matrices. Our example for this section, the 𝑋 gate, is represented by the following matrix:

$$\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$

You may recall that the state of a qubit is represented by a vector of size 2. You can apply a gate to a qubit by multiplying the gate's matrix by the qubit's state vector. The result will be another vector, representing the new state of the qubit. For example, applying the 𝑋 gate to the computational basis states looks like this:

$$X\ket{0} =
\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
\begin{bmatrix} 1 \\ 0 \end{bmatrix} =
\begin{bmatrix} 0 \cdot 1 + 1 \cdot 0 \\ 1 \cdot 1 + 0 \cdot 0 \end{bmatrix} =
\begin{bmatrix} 0 \\ 1 \end{bmatrix}$$
$$X\ket{1} =
\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
\begin{bmatrix} 0 \\ 1 \end{bmatrix} =
\begin{bmatrix} 0 \cdot 0 + 1 \cdot 1 \\ 1 \cdot 0 + 0 \cdot 1 \end{bmatrix} =
\begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

The general case :
$$\ket{\psi} = \alpha\ket{0} + \beta\ket{1}$$
$$X\ket{\psi} =
\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
\begin{bmatrix} \alpha \\ \beta \end{bmatrix} =
\begin{bmatrix} 0 \cdot \alpha + 1 \cdot \beta \\ 1 \cdot \alpha + 0 \cdot \beta \end{bmatrix} =
\begin{bmatrix} \beta \\ \alpha \end{bmatrix}$$

> Reminder of what  |0⟩, |1⟩, and |𝜓⟩ mean [[The Qubit#The Dirac Notation|here]]

Quantum gates are represented by matrices, just like quantum states are represented by vectors. Because this is the most common way to represent quantum gates, the terms "gate" and "gate matrix" will be used interchangeably.

Applying several quantum gates in sequence is equivalent to performing several of these multiplications. For example, if you have gates 𝐴 and 𝐵 and a qubit in state |𝜓⟩, the result of applying 𝐴 followed by 𝐵 to that qubit would be 𝐵(𝐴|𝜓⟩) (the gate closest to the qubit state gets applied first). Matrix multiplication is associative, so this is equivalent to multiplying the 𝐵 matrix by the 𝐴 matrix, producing a compound gate of the two, and then applying that to the qubit: (𝐵𝐴)|𝜓⟩.

All quantum gates are reversible - there is another gate which will undo any given gate's transformation, returning the qubit to its original state. This means that when dealing with quantum gates, information about qubit states is never lost, as opposed to classical logic gates, some of which destroy information. Quantum gates are represented by unitary matrices, so the inverse of a gate is its adjoint; these terms are also used interchangeably in quantum computing.

## Effects on Basis States
---

There is a simple way to find out what a gate does to the two computational basis states |0⟩ and |1⟩. Consider an arbitrary gate:

$$A = \begin{bmatrix} \epsilon & \zeta \\ \eta & \mu \end{bmatrix}$$

Watch what happens when we apply it to these states:

$$A\ket{0} =
\begin{bmatrix} \epsilon & \zeta \\ \eta & \mu \end{bmatrix}
\begin{bmatrix} 1 \\ 0 \end{bmatrix} =
\begin{bmatrix} \epsilon \cdot 1 + \zeta \cdot 0 \\ \eta \cdot 1 + \mu \cdot 0 \end{bmatrix} =
\begin{bmatrix} \epsilon \\ \eta \end{bmatrix} = \epsilon\ket{0} + \eta\ket{1}$$
$$A\ket{1} =
\begin{bmatrix} \epsilon & \zeta \\ \eta & \mu \end{bmatrix}
\begin{bmatrix} 0 \\ 1 \end{bmatrix} =
\begin{bmatrix} \epsilon \cdot 0 + \zeta \cdot 1 \\ \eta \cdot 0 + \mu \cdot 1 \end{bmatrix} =
\begin{bmatrix} \zeta \\ \mu \end{bmatrix} = \zeta\ket{0} + \mu\ket{1}$$

Notice that applying the gate to the |0⟩ state transforms it into the state written as the first column of the gate's matrix. Likewise, applying the gate to the |1⟩ state transforms it into the state written as the second column. This holds true for any quantum gate, including, of course, the 𝑋 gate:

$$X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$
$$X\ket{0} = \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \ket{1}$$
$$X\ket{1} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \ket{0}$$

Once you understand how a gate affects the computational basis states, you can easily find how it affects any state. Recall that any qubit state vector can be written as a linear combination of the basis states:

$$\ket{\psi} = \begin{bmatrix} \alpha \\ \beta \end{bmatrix} = \alpha\ket{0} + \beta\ket{1}$$

Because matrix multiplication distributes over addition, once you know how a gate affects those two basis states, you can calculate how it affects any state:

$$X\ket{\psi} = X\big(\alpha\ket{0} + \beta\ket{1}\big) = X\big(\alpha\ket{0}\big) + X\big(\beta\ket{1}\big) = \alpha X\ket{0} + \beta X\ket{1} = \alpha\ket{1} + \beta\ket{0}$$

That is, applying a gate to a qubit in superposition is equivalent to applying that gate to the basis states that make up that superposition and adding the results with appropriate weights.

## Ket-Bra Representation
---

I consider this more of a math topic so I will place it in its own little note under `Azure Quantum > Math > Computer Notes > Ket-Bra` or you can view it [[Ket-Bra|here]]. Though it's important to stop reading this note, go read ket-bra and then return here. I will put a redirect link to this note at the end of the ket-bra theory.

## Identity Gate
---
The identity gate is mostly here for completeness, at least for now. It will come in handy when dealing with multi-qubit systems and multi-qubit gates. It is represented by the identity matrix, and does not affect the state of the qubit.


![[Pasted image 20240705164936.png]]

## Hadamard Gate
---
The **Hadamard** gate is an extremely important quantum gate. Unlike the previous gates, applying the Hadamard gate to a qubit in a computational basis state puts that qubit into a superposition. Like the Pauli gates, the Hadamard gate is self-adjoint, meaning that each one is its own inverse, 𝑋𝑋=𝐼.

![[Pasted image 20240705170325.png]]

> As a reminder $e^{i\pi/4} = \frac{1}{\sqrt2} (1 + i)$ and $e^{-i\pi/4} = \frac{1}{\sqrt2} (1 - i)$. This is an application of Euler's formula, $e^{i\theta} = \cos \theta + i\sin \theta$, where 𝜃 is measured in radians. See this [Wikipedia article](https://en.wikipedia.org/wiki/Euler%27s_formula) for an explanation of Euler's formula and/or [this video](https://youtu.be/v0YEaeIClKY) for a more intuitive explanation.

### Basis change
---

**Input**: A qubit in state |𝜓⟩=𝛼|0⟩+𝛽|1⟩.

**Goal**: Change the state of the qubit as follows:

- If the qubit is in state |0⟩, change its state to $\ket{+} = \frac{1}{\sqrt{2}} \big(\ket{0} + \ket{1}\big)$.
- If the qubit is in state |1⟩, change its state to $\ket{-} = \frac{1}{\sqrt{2}} \big(\ket{0} - \ket{1}\big)$.
- If the qubit is in superposition, change its state according to the effect on basis vectors.

```C#
namespace Kata {

operation BasisChange (q : Qubit) : Unit is Adj + Ctl {
	H(q);
	DumpMachine()
	}
}
```

#### ==Solution== 

We can recognize that the Hadamard gate changes states |0⟩ and |1⟩ to |+⟩ and |−⟩, respectively, and vice versa.

As a reminder, the Hadamard gate is defined by the following matrix:

$$\frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1 \\1 & -1\end{bmatrix}$$

For example, we can work out 𝐻|1⟩ as follows:

$$H\ket{1}=
\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\1 & -1\end{bmatrix}
\begin{bmatrix} 0\\ 1\end{bmatrix}=
\frac{1}{\sqrt{2}}\begin{bmatrix}1 \cdot 0 + 1 \cdot 1 \\1 \cdot 0 + (-1) \cdot 1\end{bmatrix}=
\frac{1}{\sqrt{2}}\begin{bmatrix}1\\ -1\\ \end{bmatrix}=
\frac{1}{\sqrt{2}} \big(\ket{0} - \ket{1}\big) = \ket{-}$$
  
Similarly, we can consider the effect of the Hadamard gate on the superposition state $\ket{\psi} = 0.6\ket{0} + 0.8\ket{1}$ (rounding the numbers to 4 decimal places):

$$H|\psi⟩ =
\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
 \begin{bmatrix} \alpha\\ \beta\\ \end{bmatrix} =
\frac{1}{\sqrt{2}}\begin{bmatrix} \alpha + \beta\\ \alpha - \beta\\ \end{bmatrix}=
0.7071\begin{bmatrix} 1.4\\ -0.2\\ \end{bmatrix} =
\begin{bmatrix}
   0.98994\\ -0.14142\\ \end{bmatrix} =
   0.9899\ket{0} - 0.1414\ket{1}$$

## Phase Shift Gates
---

The next two gates are known as phase shift gates. They apply a phase to the |1⟩ state, and leave the |0⟩ state unchanged.
![[Pasted image 20240707101802.png]]

> Notice that applying the <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>T</mi>
</math> gate twice is equivalent to applying the <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>S</mi>
</math> gate, and applying the <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>S</mi>
</math> gate twice is equivalent to applying the <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>Z</mi>
</math> gate :
><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>T</mi>
    <mn>2</mn>
  </msup>
  <mo>=</mo>
  <mi>S</mi>
  </math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>S</mi>
    <mn>2</mn>
  </msup>
  <mo>=</mo>
  <mi>Z</mi>
</math>
 
## Rotation Gates
---

The next few gates are parametrized: their exact behaviour depends on a numeric parameter - an angle 𝜃, given in radians. These gates are the 𝑋 rotation gate <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>R</mi>
    <mi>x</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>&#x3B8;</mi>
  <mo stretchy="false">)</mo>
</math>, 𝑌 rotation gate <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>R</mi>
    <mi>y</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>&#x3B8;</mi>
  <mo stretchy="false">)</mo>
</math>, 𝑍 rotation gate 𝑅𝑧(𝜃), and the arbitrary phase gate <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>R</mi>
    <mn>1</mn>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>&#x3B8;</mi>
  <mo stretchy="false">)</mo>
</math>. Note that for the first three gates the parameter 𝜃 is multiplied by 12 within the gate's matrix.

>These gates are known as rotation gates, because they represent rotations around various axes on the Bloch sphere. The Bloch sphere is a way of representing the qubit states visually, mapping them onto the surface of a sphere. Unfortunately, this visualization isn't very useful beyond single-qubit states, which is why we have opted not to go into details in this kata. If you are curious about it, you can learn more in [this Wikipedia article](https://en.wikipedia.org/wiki/Bloch_sphere).


![[Pasted image 20240707103351.png]]

You have already encountered some special cases of the 𝑅1 gate:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>T</mi>
  <mo>=</mo>
  <msub>
    <mi>R</mi>
    <mn>1</mn>
  </msub>
  <mo stretchy="false">(</mo>
  <mfrac>
    <mi>&#x3C0;</mi>
    <mn>4</mn>
  </mfrac>
  <mo stretchy="false">)</mo>
  </math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>S</mi>
  <mo>=</mo>
  <msub>
    <mi>R</mi>
    <mn>1</mn>
  </msub>
  <mo stretchy="false">(</mo>
  <mfrac>
    <mi>&#x3C0;</mi>
    <mn>2</mn>
  </mfrac>
  <mo stretchy="false">)</mo>
  </math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>Z</mi>
  <mo>=</mo>
  <msub>
    <mi>R</mi>
    <mn>1</mn>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>&#x3C0;</mi>
  <mo stretchy="false">)</mo>
</math>
In addition, this gate is closely related to the 𝑅𝑧 gate: applying 𝑅1 gate is equivalent to applying the $R_z$ gate, and then applying a global phase:

$$R_1(\theta) = e^{i\theta/2}R_z(\theta)$$


In addition, the rotation gates are very closely related to their respective Pauli gates:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>X</mi>
  <mo>=</mo>
  <mi>i</mi>
  <msub>
    <mi>R</mi>
    <mi>x</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>&#x3C0;</mi>
  <mo stretchy="false">)</mo>
  </math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>Y</mi>
  <mo>=</mo>
  <mi>i</mi>
  <msub>
    <mi>R</mi>
    <mi>y</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>&#x3C0;</mi>
  <mo stretchy="false">)</mo>
  </math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>Z</mi>
  <mo>=</mo>
  <mi>i</mi>
  <msub>
    <mi>R</mi>
    <mi>z</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>&#x3C0;</mi>
  <mo stretchy="false">)</mo>
</math>
### Prepare Rotated State
---
During this section, this kata was proposed and I found it to be really difficult and a lot of the reasoning behind it wasn't very clear in it's solution. `ArcTan2()`'s documentation is really lackluster and unless you are really comfortable with trigonometry this one was tough.

**Inputs:**

1. Real numbers 𝛼 and 𝛽 such that $\alpha^2 + \beta^2 = 1$.
2. A qubit in state |0⟩.

**Goal:** Use a rotation gate to transform the qubit into state 𝛼|0⟩−𝑖𝛽|1⟩.

#### ==Solution==
---

We use the rotation gate $R_x(\theta)$. This gate turns the state |0⟩ into $R_x(\theta)\ket{0} = \cos\frac{\theta}{2}\ket{0} - i\sin\frac{\theta}{2}\ket{1}$. This is similar to the state we need. We just need to find an angle 𝜃 such that $\cos\frac{\theta}{2}=\alpha$ and $\sin\frac{\theta}{2}=\beta$ We can use these two equations to solve for 𝜃:$\theta = 2\arctan\frac{\beta}{\alpha}$ 

> (_Note: It is given that_ $\alpha^2 + \beta^2=1$, hence the required gate is $R_x(2\arctan\frac{\beta}{\alpha})$, which in matrix form is : 

$$\begin{bmatrix} \alpha & -i\beta \\ -i\beta & \alpha \end{bmatrix}$$

This gate turns $\ket{0} = \begin{bmatrix} 1 \\ 0\end{bmatrix}$ into $\begin{bmatrix} \alpha & -i\beta \\ -i\beta & \alpha \end{bmatrix} \begin{bmatrix} 1 \\ 0\end{bmatrix} = \begin{bmatrix} \alpha \\ -i\beta \end{bmatrix} = \alpha\ket{0} -i\beta\ket{1}$

The answer is :

```C#
namespace Kata {  
    open Microsoft.Quantum.Math;  
  
    operation PrepareRotatedState (alpha : Double, beta : Double, q : Qubit) : Unit is Adj+Ctl {  
        let phi = ArcTan2(beta, alpha);  
        Rx(2.0 * phi, q);  
    }  
}
```

> Basically and put more simply, sin applied to `ArcTan2(b/a)` will equal *beta* and cos applied to `ArcTan2(b/a)` will equal *alpha*. When inputed into Rx() this equals the transformation matrix above which when applied to a qubit in state 0 gives us our desired state. This [phind response](https://www.phind.com/search?cache=j893o3aac2og1i8zbyhzdhtx) helps.
## Pauli Gates
---
This section introduces some of the common single-qubit gates, including their matrix form, their ket-bra decomposition, and a brief "cheatsheet" listing their effect on some common qubit states.

You can use a tool called [Quirk](https://algassert.com/quirk) to visualize how these gates interact with various qubit states.

This section relies on the following notation:

![[Pasted image 20240704183216.png]]

The Pauli gates, named after [Wolfgang Pauli](https://en.wikipedia.org/wiki/Wolfgang_Pauli), are based on the so-called **Pauli matrices**, 𝑋, 𝑌 and 𝑍. All three Pauli gates are **self-adjoint**, meaning that each one is its own inverse, 𝑋𝑋=𝐼.

![[Pasted image 20240704183904.png]]
>The 𝑋 gate is sometimes referred to as the **bit flip** gate, or the **NOT** gate, because it acts like the classical NOT gate on the computational basis.
>
>The 𝑍 gate is sometimes referred to as the **phase flip** gate.

Here are several properties of the Pauli gates that are easy to verify and convenient to remember:

- Different Pauli gates *anti-commute*:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>X</mi>
  <mi>Z</mi>
  <mo>=</mo>
  <mo>&#x2212;</mo>
  <mi>Z</mi>
  <mi>X</mi>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
	  <mi>Y</mi>
	  <mi>Z</mi>
	  <mo>=</mo>
	  <mo>&#x2212;</mo>
	  <mi>Z</mi>
	  <mi>Y</mi>
  </math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
 <mi>X</mi>
  <mi>Y</mi>
  <mo>=</mo>
  <mo>&#x2212;</mo>
  <mi>Y</mi>
  <mi>X</mi>
</math>
- A product of any two Pauli Gates equals the third gate, with an extra i (or -i) phase:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>X</mi>
  <mi>Y</mi>
  <mo>=</mo>
  <mi>i</mi>
  <mi>Z</mi>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>Y</mi>
  <mi>Z</mi>
  <mo>=</mo>
  <mi>i</mi>
  <mi>X</mi>
</math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>Z</mi>
  <mi>X</mi>
  <mo>=</mo>
  <mi>i</mi>
  <mi>Y</mi>
</math>
- A product of all three Pauli gates equals identity (with an extra i phase):
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>X</mi>
  <mi>Y</mi>
  <mi>Z</mi>
  <mo>=</mo>
  <mi>i</mi>
  <mi>I</mi>
</math>

> This concludes this section, you can either keep reading the section pertaining to applying this theory in Q# or continue reading over at the [[Multi-Qubit Systems|Multi-Qubit Systems]] section.
### Pauli Gates in Q#`
---

The following example contains code demonstrating how to apply gates in Q#. It sets up a series of quantum states, and then shows the result of applying the 𝑋 gate to each one.

A qubit state in Q# cannot be directly assigned or accessed. The same logic is extended to quantum gates: applying a gate to a qubit modifies the internal state of that qubit, but doesn't return the resulting state of the qubit. This is why we never assign the output of these gates to any variables in this demo - they don't produce any output.

The same principle applies to applying several gates in a row to a qubit. In the mathematical notation, applying an 𝑋 gate followed by a 𝑍 gate to a state |𝜓⟩ is denoted as 𝑍(𝑋(|𝜓⟩)), because the result of applying a gate to a state is another state. In Q#, applying a gate doesn't return anything, so you can't use its output as an input to another gate - something like `Z(X(q))` will not produce the expected result. Instead, to apply several gates to the same qubit, you need to call them separately in the order in which they are applied:

```C#
X(q);
Z(q);
```

All the basic gates we will be covering in this section are part of the Intrinsic namespace. We're also using the function DumpMachine to print the state of the quantum simulator (this was only available on Microsoft's online simulator).

```C#
namespace Demo {
    // To use elements from a namespace, you need to use the `open` keyword to
    // access them.
    open Microsoft.Quantum.Diagnostics;

    @EntryPoint()
    operation PauliGatesUsage () : Unit {
        // This allocates a qubit for us to work with.
        use q = Qubit();

        // This will put the qubit into an uneven superposition |𝜓❭, where the
        // amplitudes of |0⟩ and |1⟩ have different absolute values.
        Ry(1.0, q);

        Message("Qubit in state |𝜓❭:");
        DumpMachine();

        // Let's apply the X gate; notice how it swaps the amplitudes of the
        // |0❭ and |1❭ basis states.
        X(q);
        Message("Qubit in state X|𝜓❭:");
        DumpMachine();

        // Applying the Z gate adds -1 relative phase to the |1❭ basis states.
        Z(q);
        Message("Qubit in state ZX|𝜓❭:");
        DumpMachine();

        // Finally, applying the Y gate returns the qubit to its original state
        // |𝜓❭, with an extra global phase of i.
        Y(q);
        Message("Qubit in state YZX|𝜓❭:");
        DumpMachine();

        // This returns the qubit to state |0❭.
        Reset(q);
    }
}

```

**output is** :
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>&#x3C8;</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mn>0.8776</mn>
  <mi>i</mi>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>0</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>+</mo>
  <mn>0.4794</mn>
  <mi>i</mi>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>1</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
</math>

#### State Flip
---

```C#
namespace Kata {
    open Microsoft.Quantum.Diagnostics;
    operation StateFlip (q : Qubit) : Unit is Adj + Ctl {
        X(q);
        DumpMachine();
    }
}
```

$$|\psi\rangle = \frac{1}{2}|0000\rangle+\frac{1}{2}|0101\rangle+\frac{1}{2}|1011\rangle+\frac{1}{2}|1110\rangle$$


#### Sign Flip 
---
**Input**: A qubit in state $\ket{\psi} = \alpha \ket{0} + \beta \ket{1}$.

**Goal** : Change the qubit state to 𝛼|0⟩−𝛽|1⟩ (i.e., flip the sign of the |1⟩ component of the superposition).

```C#
namespace Kata {
    open Microsoft.Quantum.Diagnostics;
    operation SignFlip (q : Qubit) : Unit is Adj + Ctl {
        // Implement your solution here...
        Z(q);
        DumpMachine();
    }
}
```

$$|\psi\rangle = \frac{1}{2}|0000\rangle+\frac{1}{2}|0101\rangle+\frac{1}{2}|1010\rangle-\frac{1}{2}|1111\rangle$$

#### Y Gate
---
**Input:** A qubit in an arbitrary state |𝜓⟩=𝛼|0⟩+𝛽|1⟩.

**Goal:** Apply the Y gate to the qubit, i.e., transform the given state into 𝑖𝛼|1⟩−𝑖𝛽|0⟩.

```C#
namespace Kata {
    open Microsoft.Quantum.Diagnostics;
    operation ApplyY(q : Qubit) : Unit is Adj + Ctl {
        Y(q);
        DumpMachine();
    }
}
```


$$|\psi\rangle = \frac{1}{2}|0000\rangle+\frac{1}{2}|0101\rangle+\frac{1}{2}i|1011\rangle-\frac{1}{2}i|1110\rangle$$

#### Sign Flip on Zero
---
**Input:** A qubit in an arbitrary state |𝜓⟩=𝛼|0⟩+𝛽|1⟩.

**Goal:** Use several Pauli gates to change the qubit state to −𝛼|0⟩+𝛽|1⟩, i.e., apply the transformation represented by the following matrix:

$$\begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$$

```C#
namespace Kata {
    operation SignFlipOnZero (q : Qubit) : Unit is Adj+Ctl {
        X(q);
        Z(q);
        X(q);
    }
}
```

$$|\psi\rangle = \frac{1}{2}|0000\rangle+\frac{1}{2}|0101\rangle-\frac{1}{2}|1010\rangle+\frac{1}{2}|1111\rangle$$

#### Global Phase `i`
---
**Input:** A qubit in an arbitrary state |𝜓⟩=𝛼|0⟩+𝛽|1⟩.

**Goal:** Use several Pauli gates to change the qubit state to 𝑖|𝜓⟩=𝑖𝛼|0⟩+𝑖𝛽|1⟩.

```C#
namespace Kata {
    open Microsoft.Quantum.Diagnostics;
    operation GlobalPhaseI(q : Qubit) : Unit is Adj + Ctl {
        Z(q);
        Y(q);
        X(q);
        DumpMachine();
    }
}
```


This concludes this section, you can continue reading over at the [[Multi-Qubit Systems|Multi-Qubit Systems]] section.