
---

> Most of these notes are provided by Microsoft [here](https://quantum.microsoft.com/en-us/experience/quantum-katas) 
> Another good source that details much found on here can be found on [this page](https://en.wikipedia.org/wiki/Determinant)


## Matrix Representation
---

Quantum gates are represented as <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mn>2</mn>
    <mi>N</mi>
  </msup>
  <mo>&#xD7;</mo>
  <msup>
    <mn>2</mn>
    <mi>N</mi>
  </msup>
</math> unitary matrices, where 𝑁 is the number of qubits the gate operates on. As a quick reminder, a unitary matrix is a square matrix whose inverse is its adjoint, thus <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>U</mi>
    <mo>&#x2217;</mo>
  </msup>
  <mi>U</mi>
  <mo>=</mo>
  <mi>U</mi>
  <msup>
    <mi>U</mi>
    <mo>&#x2217;</mo>
  </msup>
  <mo>=</mo>
  <mi>U</mi>
  <msup>
    <mi>U</mi>
    <mrow data-mjx-texclass="ORD">
      <mo>&#x2212;</mo>
      <mn>1</mn>
    </mrow>
  </msup>
  <mo>=</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="double-struck">I</mi>
  </mrow>
</math>. Single-qubit gates are represented by 2×2 matrices. Our example for this section, the 𝑋 gate, is represented by the following matrix:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mn>0</mn>
        </mtd>
        <mtd>
          <mn>1</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mn>0</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
You may recall that the state of a qubit is represented by a vector of size 2. You can apply a gate to a qubit by multiplying the gate's matrix by the qubit's state vector. The result will be another vector, representing the new state of the qubit. For example, applying the 𝑋 gate to the computational basis states looks like this:

![[Pasted image 20240629162551.png]]
The general case :
![[Pasted image 20240629162642.png]]
> Reminder of what  |0⟩, |1⟩, and |𝜓⟩ mean [[The Qubit#The Dirac Notation|here]]

Quantum gates are represented by matrices, just like quantum states are represented by vectors. Because this is the most common way to represent quantum gates, the terms "gate" and "gate matrix" will be used interchangeably.

Applying several quantum gates in sequence is equivalent to performing several of these multiplications. For example, if you have gates 𝐴 and 𝐵 and a qubit in state |𝜓⟩, the result of applying 𝐴 followed by 𝐵 to that qubit would be 𝐵(𝐴|𝜓⟩) (the gate closest to the qubit state gets applied first). Matrix multiplication is associative, so this is equivalent to multiplying the 𝐵 matrix by the 𝐴 matrix, producing a compound gate of the two, and then applying that to the qubit: (𝐵𝐴)|𝜓⟩.

All quantum gates are reversible - there is another gate which will undo any given gate's transformation, returning the qubit to its original state. This means that when dealing with quantum gates, information about qubit states is never lost, as opposed to classical logic gates, some of which destroy information. Quantum gates are represented by unitary matrices, so the inverse of a gate is its adjoint; these terms are also used interchangeably in quantum computing.

## Effects on Basis States
---

There is a simple way to find out what a gate does to the two computational basis states |0⟩ and |1⟩. Consider an arbitrary gate:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>A</mi>
  <mo>=</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mi>&#x3F5;</mi>
        </mtd>
        <mtd>
          <mi>&#x3B6;</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>&#x3B7;</mi>
        </mtd>
        <mtd>
          <mi>&#x3BC;</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
Watch what happens when we apply it to these states:

![[Pasted image 20240629163401.png]]
Notice that applying the gate to the |0⟩ state transforms it into the state written as the first column of the gate's matrix. Likewise, applying the gate to the |1⟩ state transforms it into the state written as the second column. This holds true for any quantum gate, including, of course, the 𝑋 gate:
![[Pasted image 20240629163615.png]]
Once you understand how a gate affects the computational basis states, you can easily find how it affects any state. Recall that any qubit state vector can be written as a linear combination of the basis states:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>&#x3C8;</mi>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
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
Because matrix multiplication distributes over addition, once you know how a gate affects those two basis states, you can calculate how it affects any state:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>X</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>&#x3C8;</mi>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>=</mo>
  <mi>X</mi>
  <mrow data-mjx-texclass="ORD">
    <mo minsize="1.2em" maxsize="1.2em">(</mo>
  </mrow>
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
  <mrow data-mjx-texclass="ORD">
    <mo minsize="1.2em" maxsize="1.2em">)</mo>
  </mrow>
  <mo>=</mo>
  <mi>X</mi>
  <mrow data-mjx-texclass="ORD">
    <mo minsize="1.2em" maxsize="1.2em">(</mo>
  </mrow>
  <mi>&#x3B1;</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>0</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mo minsize="1.2em" maxsize="1.2em">)</mo>
  </mrow>
  <mo>+</mo>
  <mi>X</mi>
  <mrow data-mjx-texclass="ORD">
    <mo minsize="1.2em" maxsize="1.2em">(</mo>
  </mrow>
  <mi>&#x3B2;</mi>
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
  <mo>=</mo>
  <mi>&#x3B1;</mi>
  <mi>X</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>0</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>+</mo>
  <mi>&#x3B2;</mi>
  <mi>X</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>=</mo>
  <mi>&#x3B1;</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>+</mo>
  <mi>&#x3B2;</mi>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>0</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
</math>
That is, applying a gate to a qubit in superposition is equivalent to applying that gate to the basis states that make up that superposition and adding the results with appropriate weights.

## Ket-Bra Representation
---

I consider this more of a math topic so I will place it in its own little note under `Azure Quantum > Math > Computer Notes > Ket-Bra` or you can view it [[Ket-Bra|here]]. Though it's important to stop reading this note, go read ket-bra and then return here. I will put a redirect link to this note at the end of the ket-bra theory.

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

![[Pasted image 20240704185853.png]]
![[Pasted image 20240704185927.png]]
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
### State Flip
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

![[Pasted image 20240704190516.png]]
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>&#x3C8;</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>0000</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>+</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>0101</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>+</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>1011</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>+</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>1110</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
</math>

### Sign Flip 
---
**Input**: A qubit in state |𝜓⟩=𝛼|0⟩+𝛽|1⟩.

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
![[Pasted image 20240704190707.png]]
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>&#x3C8;</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>0000</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>+</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>0101</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>+</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>1010</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>&#x2212;</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>1111</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
</math>
### Y Gate
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

![[Pasted image 20240704190828.png]]
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>&#x3C8;</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>0000</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>+</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>0101</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>+</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mi>i</mi>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>1011</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>&#x2212;</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mi>i</mi>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>1110</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
</math>
### Sign Flip on Zero
---
**Input:** A qubit in an arbitrary state |𝜓⟩=𝛼|0⟩+𝛽|1⟩.

**Goal:** Use several Pauli gates to change the qubit state to −𝛼|0⟩+𝛽|1⟩, i.e., apply the transformation represented by the following matrix:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mo>&#x2212;</mo>
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

```C#
namespace Kata {
    operation SignFlipOnZero (q : Qubit) : Unit is Adj+Ctl {
        X(q);
        Z(q);
        X(q);
    }
}
```
![[Pasted image 20240704191739.png]]
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mi>&#x3C8;</mi>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>0000</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>+</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>0101</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>&#x2212;</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>1010</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
  <mo>+</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo data-mjx-texclass="ORD" stretchy="false">|</mo>
  <mn>1111</mn>
  <mo fence="false" stretchy="false">&#x27E9;</mo>
</math>
### Global Phase `i`
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
![[Pasted image 20240705164036.png]]

## Identity Gate
---
The identity gate is mostly here for completeness, at least for now. It will come in handy when dealing with multi-qubit systems and multi-qubit gates. It is represented by the identity matrix, and does not affect the state of the qubit.


![[Pasted image 20240705164936.png]]

## Hadamard Gate
---
The **Hadamard** gate is an extremely important quantum gate. Unlike the previous gates, applying the Hadamard gate to a qubit in a computational basis state puts that qubit into a superposition. Like the Pauli gates, the Hadamard gate is self-adjoint, meaning that each one is its own inverse, 𝑋𝑋=𝐼.

![[Pasted image 20240705170325.png]]

> As a reminder <math xmlns="http://www.w3.org/1998/Math/MathML">
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
  <mo stretchy="false">(</mo>
  <mn>1</mn>
  <mo>+</mo>
  <mi>i</mi>
  <mo stretchy="false">)</mo>
</math> and <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mo>&#x2212;</mo>
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
  <mo stretchy="false">(</mo>
  <mn>1</mn>
  <mo>&#x2212;</mo>
  <mi>i</mi>
  <mo stretchy="false">)</mo>
</math>. This is an application of Euler's formula, <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math>, where 𝜃 is measured in radians. See this [Wikipedia article](https://en.wikipedia.org/wiki/Euler%27s_formula) for an explanation of Euler's formula and/or [this video](https://youtu.be/v0YEaeIClKY) for a more intuitive explanation.

### Basis change
---

**Input**: A qubit in state |𝜓⟩=𝛼|0⟩+𝛽|1⟩.

**Goal**: Change the state of the qubit as follows:

- If the qubit is in state |0⟩, change its state to <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math>.
- If the qubit is in state |1⟩, change its state to <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math>.
- If the qubit is in superposition, change its state according to the effect on basis vectors.

```C#
namespace Kata {

operation BasisChange (q : Qubit) : Unit is Adj + Ctl {
	H(q);
	DumpMachine()
	}
}
```

We can recognize that the Hadamard gate changes states |0⟩ and |1⟩ to |+⟩ and |−⟩, respectively, and vice versa.

As a reminder, the Hadamard gate is defined by the following matrix:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
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
          <mn>1</mn>
        </mtd>
        <mtd>
          <mo>&#x2212;</mo>
          <mn>1</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
For example, we can work out 𝐻|1⟩ as follows:

![[Pasted image 20240707095821.png]]

  
Similarly, we can consider the effect of the Hadamard gate on the superposition state |𝜓⟩=0.6|0⟩+0.8|1⟩ (rounding the numbers to 4 decimal places):

![[Pasted image 20240707095902.png]]
![[Pasted image 20240707095942.png]]

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
In addition, this gate is closely related to the 𝑅𝑧 gate: applying 𝑅1 gate is equivalent to applying the 𝑅𝑧 gate, and then applying a global phase:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mi>R</mi>
    <mn>1</mn>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>&#x3B8;</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <msup>
    <mi>e</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mi>&#x3B8;</mi>
      <mrow data-mjx-texclass="ORD">
        <mo>/</mo>
      </mrow>
      <mn>2</mn>
    </mrow>
  </msup>
  <msub>
    <mi>R</mi>
    <mi>z</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>&#x3B8;</mi>
  <mo stretchy="false">)</mo>
</math>
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
During this section, this kata was proposed and I found it to be really difficult and a lot of the reasoning behind it wasn't very clear in it's solution. `ArcTan2()`'s documentation is really lacklustre and unless you are really comfortable with trigonometry this one was tough.

**Inputs:**

1. Real numbers 𝛼 and 𝛽 such that <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>&#x3B1;</mi>
    <mn>2</mn>
  </msup>
  <mo>+</mo>
  <msup>
    <mi>&#x3B2;</mi>
    <mn>2</mn>
  </msup>
  <mo>=</mo>
  <mn>1</mn>
</math>.
2. A qubit in state |0⟩.

**Goal:** Use a rotation gate to transform the qubit into state 𝛼|0⟩−𝑖𝛽|1⟩.

===Solution===:

We use the rotation gate <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>R</mi>
    <mi>x</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>&#x3B8;</mi>
  <mo stretchy="false">)</mo>
</math>. This gate turns the state |0⟩ into <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>R</mi>
    <mi>x</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>&#x3B8;</mi>
  <mo stretchy="false">)</mo>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>0</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>=</mo>
  <mi>cos</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mfrac>
    <mi>&#x3B8;</mi>
    <mn>2</mn>
  </mfrac>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>0</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
  <mo>&#x2212;</mo>
  <mi>i</mi>
  <mi>sin</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mfrac>
    <mi>&#x3B8;</mi>
    <mn>2</mn>
  </mfrac>
  <mrow data-mjx-texclass="ORD">
    <mo data-mjx-texclass="ORD" fence="false" stretchy="false">|</mo>
    <mrow data-mjx-texclass="ORD">
      <mn>1</mn>
    </mrow>
    <mo fence="false" stretchy="false">&#x27E9;</mo>
  </mrow>
</math>. This is similar to the state we need. We just need to find an angle 𝜃 such that <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>cos</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mfrac>
    <mi>&#x3B8;</mi>
    <mn>2</mn>
  </mfrac>
  <mo>=</mo>
  <mi>&#x3B1;</mi>
</math> and <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>sin</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mfrac>
    <mi>&#x3B8;</mi>
    <mn>2</mn>
  </mfrac>
  <mo>=</mo>
  <mi>&#x3B2;</mi>
</math> We can use these two equations to solve for 𝜃: 

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>&#x3B8;</mi>
  <mo>=</mo>
  <mn>2</mn>
  <mi>arctan</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mfrac>
    <mi>&#x3B2;</mi>
    <mi>&#x3B1;</mi>
  </mfrac>
</math>
(_Note: It is given that_<math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>&#x3B1;</mi>
    <mn>2</mn>
  </msup>
  <mo>+</mo>
  <msup>
    <mi>&#x3B2;</mi>
    <mn>2</mn>
  </msup>
  <mo>=</mo>
  <mn>1</mn>
</math>, hence the required gate is <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>R</mi>
    <mi>x</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mn>2</mn>
  <mi>arctan</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mfrac>
    <mi>&#x3B2;</mi>
    <mi>&#x3B1;</mi>
  </mfrac>
  <mo stretchy="false">)</mo>
</math>, which in matrix form is : 

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mtable columnspacing="1em" rowspacing="4pt">
      <mtr>
        <mtd>
          <mi>&#x3B1;</mi>
        </mtd>
        <mtd>
          <mo>&#x2212;</mo>
          <mi>i</mi>
          <mi>&#x3B2;</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mo>&#x2212;</mo>
          <mi>i</mi>
          <mi>&#x3B2;</mi>
        </mtd>
        <mtd>
          <mi>&#x3B1;</mi>
        </mtd>
      </mtr>
    </mtable>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
</math>
![[Pasted image 20240717135940.png]]

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

This concludes this section, you can continue reading over at the [[Multi-Qubit Systems|Multi-Qubit Systems]] section.

