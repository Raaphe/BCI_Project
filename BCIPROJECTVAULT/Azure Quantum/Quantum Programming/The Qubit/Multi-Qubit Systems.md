
---

This section you will need to know these concepts ;

- [[Qubit Gates|Basic Single-Qubit Gates]]
- [[Matrices#Tensor Products|The Concept of Tensor Products]]

---

A multi-qubit system is a collection of multiple qubits, treated as a single system. 

Let's start by examining a system of two classical bits. Each bit can be in two states: 0 and 1. Therefore, a system of two bits can be in four different states: 00, 01, 10, and 11. Generally, a system of 𝑁 classical bits can be in any of the <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mn>2</mn>
    <mi>N</mi>
  </msup>
</math> states.

A system of 𝑁 qubits can also be in any of the <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mn>2</mn>
    <mi>N</mi>
  </msup>
</math> classical states, but, unlike the classical bits, it can also be in a **superposition** of all these states.

Similarly to single-qubit systems, a state of an 𝑁-qubit system can be represented as a complex vector of size <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mn>2</mn>
    <mi>N</mi>
  </msup>
</math>:

$$\begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_{2^N-1}\end{bmatrix}$$

## Basis States
---

Similarly to single-qubit systems, multi-qubit systems have their own sets of basis states. The computational basis for an 𝑁-qubit system is a set of <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mn>2</mn>
    <mi>N</mi>
  </msup>
</math> vectors, in each of which with one element equals 1, and the other elements equal 0.

For example, this is the **computational basis** for a two-qubit system:

$$\begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix},
\begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix},
\begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix},
\begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \end{bmatrix}$$

It is easy to see that these vectors form an orthonormal basis. Note that each of these basis states can be represented as a tensor product of some combination of single-qubit basis states:

![[Pasted image 20240717143452.png]]

Any two-qubit system can be expressed as some linear combination of those tensor products of single-qubit basis states.

Similar logic applies to systems of more than two qubits. In general case,

$$\begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_{2^N-1} \end{bmatrix} =
x_0 \begin{bmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix} +
x_1 \begin{bmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix} + \dotsb +
x_{2^N-1} \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$

The coefficients of the basis vectors define how "close" is the system state to the corresponding basis vector.

## Bell Basis States
---

> Just like with single-qubit systems, there exist other orthonormal bases states for multi-qubit systems. An example for a two-qubit system is the **Bell basis**:
> $$\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 0 \\ 0 \\ 1 \end{bmatrix},
\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 0 \\ 0 \\ -1 \end{bmatrix},
\frac{1}{\sqrt{2}}\begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \end{bmatrix},
\frac{1}{\sqrt{2}}\begin{bmatrix} 0 \\ 1 \\ -1 \\ 0 \end{bmatrix}$$
>  You can check that these vectors are normalized, and orthogonal to each other, and that any two-qubit state can be expressed as a linear combination of these vectors. The vectors of Bell basis, however, can not be represented as tensor products of single-qubit basis states.


## Separable States
---


Sometimes the global state of a multi-qubit system can be separated into the states of individual qubits or subsystems. To do this, you would express the vector state of the global system as a tensor product of the vectors representing each individual qubit/subsystem. Here is an example of a two-qubit state:

$$\begin{bmatrix} \frac{1}{\sqrt{2}} \\ 0 \\ \frac{1}{\sqrt{2}} \\ 0 \end{bmatrix} =
\begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} \otimes \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

You can see that the first qubit is in state $\frac{1}{\sqrt{2}}\big(\ket{0} + \ket{1}\big)$ and the second qubit is in state |0⟩. The multi-qubit states that allow such representation are known as **separable states**, or product states, because you can separate the global state into the tensor product of individual subsystems.

### Example 
---

Show that the state is separable:

$$\frac{1}{2} \begin{bmatrix} 1 \\ i \\ -i \\ 1 \end{bmatrix} =
\begin{bmatrix} ? \\ ? \end{bmatrix} \otimes \begin{bmatrix} ? \\ ? \end{bmatrix}$$

===Solution=== : To separate the state into a tensor product of two single-qubit states, we need to represent it in the following way:

$$\begin{bmatrix} \alpha \gamma \\ \alpha \delta \\ \beta \gamma \\ \beta \delta \end{bmatrix} =
\begin{bmatrix} \alpha \\ \beta \end{bmatrix} \otimes \begin{bmatrix} \gamma \\ \delta \end{bmatrix}$$

This brings us to a system of equations:

$$\begin{cases}
\alpha\gamma = \frac{1}{2} \\ \alpha\delta = \frac{i}{2} \\ \beta \gamma = \frac{-i}{2} \\ \beta \delta = \frac{1}{2}
\end{cases}$$
$$\alpha = \frac{1}{\sqrt2}, \beta = \frac{-i}{\sqrt2}, \gamma = \frac{1}{\sqrt2}, \delta = \frac{i}{\sqrt2}$$
$$\frac{1}{2} \begin{bmatrix} 1 \\ i \\ -i \\ 1 \end{bmatrix} =
\frac{1}{\sqrt2} \begin{bmatrix} 1 \\ -i \end{bmatrix} \otimes \frac{1}{\sqrt2} \begin{bmatrix} 1 \\ i \end{bmatrix}$$

## Entanglement
---

Sometimes, quantum states cannot be written as individual qubit states. Quantum systems that are not separable are called **entangled** systems. If a state can be written as the product state of the individual subsystems, that state is not entangled.

Entanglement is a quantum correlation, which is very different from classical correlations. In entanglement, the state of the subsystems isn't determined, and you can talk only about the probabilities associated with the outcomes. The global system must be considered as one.

> For example, every state in the Bell basis is an entangled state.

Entanglement is a huge part of what makes quantum computing so powerful. It allows us to link the qubits so that they stop behaving like individuals and start behaving like a large, more complex system. In entangled systems, measuring one of the qubits modifies the state of the other qubits, and tells us something about their state.

For example, consider two qubits 𝐴 and 𝐵 in superposition such that the state of the global system is

$$\ket{\psi}_{AB} = \frac{1}{\sqrt2}\ket{00} + \frac{1}{\sqrt2}\ket{11}$$

In such a state, only two outcomes are possible when you measure the state of both qubits in the standard basis: |00⟩ and |11⟩. Notice that each outcome has the same probability of $\frac{1}{2}$. There's zero probability of obtaining |01⟩ and |10⟩. If you measure the first qubit and you get that it is in |0⟩ state, then you can be positive that the second qubit is also in |0⟩ state, even without measuring it. The measurement outcomes are correlated, and the qubits are entangled.

This property is used extensively in many quantum algorithms.

## Dirac Notation
---

Just like with single qubits, Dirac notation provides a useful shorthand for writing down states of multi-qubit systems.

As we've seen earlier, multi-qubit systems have their own canonical bases, and the basis states can be represented as tensor products of single-qubit basis states. Any multi-qubit system can be represented as a linear combination of these basis states:

$$\begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \end{bmatrix} =
x_0\begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix} +
x_1\begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix} +
x_2\begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix} +
x_3\begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \end{bmatrix} =
x_0\ket{0} \otimes \ket{0} +
x_1\ket{0} \otimes \ket{1} +
x_2\ket{1} \otimes \ket{0} +
x_3\ket{1} \otimes \ket{1}$$

To simplify this, tensor products of basis states have their own notation:
$$\ket{0} \otimes \ket{0} = \ket{00}$$
$$\ket{0} \otimes \ket{1} = \ket{01}$$
$$\ket{1} \otimes \ket{0} = \ket{10}$$
$$\ket{1} \otimes \ket{1} = \ket{11}$$
$$\ket{0} \otimes \ket{0} \otimes \ket{0} = \ket{000}$$
And so on.

Or, more generally:
$$\ket{i_0} \otimes \ket{i_1} \otimes \dotsb \otimes \ket{i_n} = \ket{i_0i_1...i_n}$$
Using this notation simplifies our example:

$$\begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \end{bmatrix} =
x_0\ket{00} + x_1\ket{01} + x_2\ket{10} + x_3\ket{11}$$
Just like with single qubits, we can put arbitrary symbols within the kets the same way variables are used in algebra. Whether a ket represents a single qubit or an entire system depends on the context. Some ket symbols have a commonly accepted usage, such as the symbols for the Bell basis:

$$\ket{\Phi^{+} = \frac{1}{\sqrt{2}}\big(\ket{00} + \ket{11}\big)}$$
$$\ket{\Phi^-} = \frac{1}{\sqrt{2}}\big(\ket{00} - \ket{11}\big)$$
$$\ket{\Psi^+} = \frac{1}{\sqrt{2}}\big(\ket{01} + \ket{10}\big)$$
$$\ket{\Psi^-} = \frac{1}{\sqrt{2}}\big(\ket{01} - \ket{10}\big)$$
### Endianness
---

In classical computing, endianness refers to the order of bits (or bytes) when representing numbers in binary. You're probably familiar with the typical way of writing numbers in binary: $0 = 0_2$, $1 = 1_2$, $2 = 10_2$, $3=11_2$, $4=100_2$, $5=101_2$, $6=110_2$, etc. This is known as **big-endian format**. In big-endian format, the _most significant_ bits come first. For example: $110_2=1⋅4+1⋅2+0⋅1=4+2=6$.

There is an alternate way of writing binary numbers - **little-endian format**. In little-endian format, the _least significant_ bits come first. For example, 2 would be written as 01, 4 as 001, and 6 as 011. To put it another way, in little endian format, the number is written backwards compared to the big-endian format.

In Dirac notation for multi-qubit systems, it's common to see integer numbers within the kets instead of bit sequences. What those numbers mean depends on the context - whether the notation used is big-endian or little-endian.

Examples with a 3 qubit system:


| Integer Ket   | $\ket{0}$   | $\ket{1}$   | $\ket{2}$   | $\ket{3}$   | $\ket{4}$   | $\ket{5}$   | $\ket{6}$   | $\ket{7}$   |
| ------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Big-endian    | $\ket{000}$ | $\ket{001}$ | $\ket{010}$ | $\ket{011}$ | $\ket{100}$ | $\ket{101}$ | $\ket{110}$ | $\ket{111}$ |
| Little-endian | $\ket{000}$ | $\ket{100}$ | $\ket{010}$ | $\ket{110}$ | $\ket{001}$ | $\ket{101}$ | $\ket{011}$ | $\ket{111}$ |
Multi-qubit quantum systems that store superpositions of numbers are often referred to as **quantum registers**.

## Multi-Qubit Systems In Q#`
---

This demo shows you how to allocate multiple qubits in Q# and examine their joint state. It uses single-qubit gates for manipulating the individual qubit states - if you need a refresher on them, please review the Single-Qubit Gates kata.

These demos use the function `DumpMachine` to print the state of the quantum simulator. When dealing with multi-qubit systems, `DumpMachine` prints information about each basis state that has a non-zero amplitude, one basis state per row, the same as it does for single-qubit systems. The basis states are represented as bit strings, one bit per the qubit allocated, with the leftmost bit corresponding to the qubit that was allocated the earliest. (If the qubits were allocated at once as an array, the leftmost bit corresponds to the first element of the array.)

```C#
namespace Kata {
    open Microsoft.Quantum.Diagnostics;

    @EntryPoint()
    operation MultiQubitSystemsDemo () : Unit {
        // This allocates an array of 2 qubits, each of them in state |0⟩.
        // The overall state of the system is |00⟩.
        use qs = Qubit[2];
        // X gate changes the first qubit into state |1⟩.
        X(qs[0]);
        Message("The system in now in state |10⟩:");
        DumpMachine();

        // This changes the second qubit into state |+⟩ = (1/sqrt(2))(|0⟩ + |1⟩).
        H(qs[1]);
        Message("The system in now in state  (1/sqrt(2))(|10⟩ + |11⟩):");
        DumpMachine();

        // This changes the first qubit into state |-⟩ = (1/sqrt(2))(|0⟩ - |1⟩)
        H(qs[0]);
        Message("The system in now in state 0.5(|00⟩ + |01⟩ - |10⟩ - |11⟩):");
        DumpMachine();

        // The next lines entangle the qubits (don't worry about what exactly they do for now).
        H(qs[1]);
        CNOT(qs[0], qs[1]);
        Message("The system in now in entangled state 0.5(|00⟩ - |11⟩):");
        DumpMachine();

        // This returns the system into state |00⟩.
        ResetAll(qs);
    }
}
```

>You might have noticed that we've been "resetting" the qubits at the end of our demos, that is, returning them to |0⟩ state. Q# requires you to return your qubits into the |0⟩ state before they are released at the end of their scope. The reason for this is entanglement.
>
>Consider running a program on a quantum computer: the number of qubits is very limited, and you want to reuse the released qubits in other parts of the program. If they are not in zero state by that time, they can potentially be still entangled with the qubits which are not yet released, thus operations you perform on them can affect the state of other parts of the program, causing erroneous and hard to debug behaviour.
>
>Resetting the qubits to zero state automatically when they go outside the scope of the block they were allocated in is dangerous as well: if they were entangled with others, measuring them to reset them can affect the state of the unreleased qubits, and thus change the results of the program - without the developer noticing this.
>
>The requirement that the qubits should be in zero state before they can be released aims to remind the developer to double-check that all necessary information has been properly extracted from the qubits, and that they are not entangled with unreleased qubits any more.

## Separable State Preparation
---

In the following exercises you will learn to prepare separable quantum states by manipulating individual qubits. You will only need knowledge from the Single-Qubit Gates kata for that.

>In each exercise, you'll be given an array of qubits to manipulate; you can access 𝑖-th element of the array `qs` as `qs[i]`. Array elements are indexed starting with 0, the first array element corresponds to the leftmost qubit in Dirac notation.

### Prepare a Basis State
---

**Input:** A two-qubit system in the basis state $\ket{00} = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}$
**Goal:** Transform the system into the basis state $\ket{11} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \end{bmatrix}$

```C#
namespace Kata {
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Diagnostics;
    operation PrepareBasisState(qs : Qubit[]) : Unit is Adj + Ctl {
        X(qs[0]);
        X(qs[1]);
        DumpMachine();
    }
}
```

### Prepare a Superposition of  Two Basis States
---

**Input:** A two-qubit system in the basis state $\ket{00} = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}$
**Goal:** Transform the system into the state $\frac{1}{\sqrt2}\big(\ket{00} - \ket{01}\big) = \frac{1}{\sqrt2}\begin{bmatrix} 1 \\ -1 \\ 0 \\ 0 \end{bmatrix}$

==Solution:==

We begin in the same state as the previous exercise:

$$\begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \otimes \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \ket{0} \otimes \ket{0}$$

The goal state can be separated as follows:

$$\frac{1}{\sqrt2} \begin{bmatrix} 1 \\ -1 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \otimes \frac{1}{\sqrt2}\begin{bmatrix} 1 \\ -1 \end{bmatrix} = \ket{0} \otimes \frac{1}{\sqrt2}\big(\ket{0} - \ket{1}\big)$$

This means that the first qubit is already in the state we want it to be, but the second qubit needs to be transformed from the $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ into $\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1\end{bmatrix}$ state.

First, we apply the **X** gate to the second qubit; this performs the following transformation:

$$X \ket{0} = \begin{bmatrix}0 & 1 \\ 1 & 0 \end{bmatrix} \cdot \begin{bmatrix}1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \ket{1}$$

Second, we apply the **H** gate to the second qubit; this transforms its state into the desired one:

$$H\ket{1} = \frac{1}{\sqrt2}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \cdot \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \frac{1}{\sqrt2}\begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

```C#
namespace Kata {
    open Microsoft.Quantum.Diagnostics;
    operation PrepareSuperposition(qs : Qubit[]) : Unit is Adj + Ctl {
        X(qs[1]);
        H(qs[1]);
        // DumpRegister(qs);
        DumpMachine();

    }
}
```

### Prepare a Superposition with Real Amplitudes
---

**Input:** A two-qubit system in the basis state $\ket{00} = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}$
**Goal:** Transform the system into the state $\frac{1}{2}\big(\ket{00} - \ket{01} + \ket{10} - \ket{11}\big) = \frac{1}{2}\begin{bmatrix} 1 \\ -1 \\ 1 \\ -1 \end{bmatrix}$

>Represent the target state as a tensor product $\frac{1}{\sqrt2}\big(\ket{0} + \ket{1}\big) \otimes \frac{1}{\sqrt2}\big(\ket{0} - \ket{1}\big) = \frac{1}{\sqrt2} \begin{bmatrix} 1 \\ 1 \end{bmatrix} \otimes \frac{1}{\sqrt2}\begin{bmatrix} 1 \\ -1 \end{bmatrix}$

```C#
namespace Kata {
    open Microsoft.Quantum.Diagnostics;
    operation PrepareWithReal(qs : Qubit[]) : Unit is Adj + Ctl {
        // Implement your solution here...
        H(qs[0]);
        X(qs[1]);
        H(qs[1]);
        DumpRegister(qs);
    }
}
```

### Prepare a Superposition with Complex Amplitudes
---

**Input:** A two-qubit system in the basis state $\ket{00} = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}$

**Goal:** Transform the system into the state $\frac{1}{2}\big(\ket{00} + e^{i\pi/4}\ket{01} + e^{i\pi/2}\ket{10} + e^{3i\pi/4}\ket{11}\big) = \frac{1}{2}\begin{bmatrix} 1 \\ e^{i\pi/4} \\ e^{i\pi/2} \\ e^{3i\pi/4} \end{bmatrix}$

Represent the target state as a tensor product $\frac{1}{\sqrt2}\big(\ket{0} + e^{i\pi/2}\ket{1}\big) \otimes \frac{1}{\sqrt2}\big(\ket{0} + e^{i\pi/4}\ket{1}\big) = \frac{1}{\sqrt2} \begin{bmatrix} 1 \\ e^{i\pi/2} \end{bmatrix} \otimes \frac{1}{\sqrt2}\begin{bmatrix} 1 \\ e^{i\pi/4} \end{bmatrix}$

```C#
namespace Kata {
    operation PrepareWithComplex(qs : Qubit[]) : Unit is Adj + Ctl {
        H(qs[0]);
        H(qs[1]);
        S(qs[0]);
        T(qs[1]);
    }
}
```

## Modifying Entangled States
---

Entangled quantum states can be manipulated using single-qubit gates. For example, each state in the Bell basis is entangled and can be transformed into another Bell state through the application of single-qubit gates. In this lesson, you'll learn how to do that. (And we will learn more about applying single-qubit gates to multi-qubit states in the next kata.)

### Bell State Change 1
---

**Input:** Two entangled qubits in Bell state $\ket{\Phi^{+}} = \frac{1}{\sqrt{2}} \big(\ket{00} + \ket{11}\big)$

**Goal:** Change the two-qubit state to $\ket{\Phi^{-}} = \frac{1}{\sqrt{2}} \big(\ket{00} - \ket{11}\big)$

#### ==Solution==

We recognize that the goal is another Bell state. In fact, it is one of the four [[Multi-Qubit Systems#Bell Basis States|Bell states]].

We remember from the Single-Qubit Gates kata that the [[Qubit Gates#Pauli Gates|Pauli Z gate]] will change the state of the |1⟩ basis state of a single qubit, so this gate seems like a good candidate for what we want to achieve. This gate leaves the sign of the |0⟩ basis state of a superposition unchanged, but flips the sign of the |1⟩ basis state of the superposition.

Don't forget that the Z gate acts on only a single qubit, and we have two here. Let's also remember how the Bell state is made up from its individual qubits.

If the two qubits are A and B, where A is `qs[0]` and B is `qs[1]`, we can write that $\ket{\Phi^{+}} = \frac{1}{\sqrt{2}} \big(\ket{0_{A}0_{B}} + \ket{1_{A}1_{B}}\big)$. If we apply the Z gate to the qubit A, it will flip the phase of the basis state $\ket{1_A}$. As this phase is in a sense spread across the entangled state, with $\ket{1_A}$ basis state being part of the second half of the superposition, this application has the effect of flipping the sign of the whole basis state $\ket{1_A1_B}$, as you can see by running the solution below.

The exact same calculations can be done if we apply Z to the qubit B, so that's another possible solution.

```C#
namespace Kata {
    operation BellStateChange1 (qs : Qubit[]) : Unit is Adj + Ctl {
        Z(qs[0]);
    }
}
```

### Bell State Change 2
---

**Input:** Two entangled qubits in Bell state $\ket{\Phi^{+}} = \frac{1}{\sqrt{2}} \big(\ket{00} + \ket{11}\big)$

**Goal:** Change the two-qubit state to $\ket{\Psi^{+}} = \frac{1}{\sqrt{2}} \big(\ket{01} + \ket{10}\big)$

#### ==Solution==

We have seen in the Single-Qubit Gates kata that the Pauli X gate flips |0⟩ to |1⟩ and vice versa, and as we seem to need some flipping of states, perhaps this gate may be of use. (Bearing in mind, of course, that the X gate operates on a single qubit).

Let's compare the starting state $\frac{1}{\sqrt{2}} \big(\ket{0_A0_B} + \ket{1_A1_B}\big)$ with the goal state $\frac{1}{\sqrt{2}} \big(\ket{1_A0_B} + \ket{0_A1_B}\big)$ term by term and see how we need to transform it to reach the goal.

Using our nomenclature from "Bell state change 1", we can now see by comparing terms that $\ket{0_{A}}$ has flipped to $\ket{1_{A}}$ to get the first term, and $\ket{1_{A}}$ has flipped to $\ket{0_{A}}$ to get the second term. This allows us to say that the correct gate to use is Pauli X, applied to `qs[0]`.

```C#
namespace Kata {
    operation BellStateChange2 (qs : Qubit[]) : Unit is Adj + Ctl {
        X(qs[0]);
    }
}
```

### Bell State Change 3
---

**Input:** Two entangled qubits in Bell state $\ket{\Phi^{+}} = \frac{1}{\sqrt{2}} \big(\ket{00} + \ket{11}\big)$

**Goal:** Change the two-qubit state, without adding a global phase, to $\ket{\Psi^{-}} = \frac{1}{\sqrt{2}} \big(\ket{01} - \ket{10}\big)$

#### ==Solution==

We remember from the Single-Qubit Gates kata that the Pauli Z gate leaves the sign of the |0⟩ component of the single qubit superposition unchanged but flips the sign of the |1⟩ component of the superposition. We have also just seen in "Bell State Change 2" how to change our input state to the state $\frac{1}{\sqrt{2}} \big(\ket{01} + \ket{10}\big)$, which is almost our goal state (disregarding the phase change for the moment). So it would seem that a combination of these two gates will be what we need here. The remaining question is in what order to apply them, and to which qubit.

First of all, which qubit? Looking back at the task "Bell state change 2", it seems clear that we need to use qubit `qs[0]`, like we did there.

Second, in what order should we apply the gates? Remember that the Pauli Z gate flips the phase of the |1⟩ component of the superposition and leaves the |0⟩ component alone. Let's experiment with applying X to `qs[0]` first. Looking at our "halfway answer" state $\frac{1}{\sqrt{2}} \big(\ket{01} + \ket{10}\big)$, we can see that if we apply the Z gate to `qs[0]`, it will leave the |0𝐴⟩ alone but flip the phase of $\ket{1_{A}}$ to $-\ket{1_{A}}$, thus flipping the phase of the |11⟩ component of our Bell state.

```C#
namespace Kata {
    operation BellStateChange3(qs : Qubit[]) : Unit is Adj + Ctl {
        X(qs[0]);
        Z(qs[0]);
    }
}
```