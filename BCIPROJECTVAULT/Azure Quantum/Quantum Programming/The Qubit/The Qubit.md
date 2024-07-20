
---
The basic building block of a classical computer is the bit - a single memory cell that is either in state 0 or in state 1. Similarly, the basic building block of a quantum computer is the quantum bit, or **qubit**. Like the classical bit, a qubit can be in state 0 or in state 1. Unlike the classical bit, however, the qubit isn't limited to just those two states - it may also be in a combination, or **superposition** of those states.

>A common misconception about quantum computing is that a qubit is always in state 1 or state 0, we just don't know which one until we "measure" it. That is not the case. A qubit in a superposition is in a linear combination of the states 0 and 1. When a qubit is measured, it is forced to collapse into one state or the other - in other words, measuring a qubit is an irreversible process that changes its initial state.


## Matrix Representation
---
The state of a qubit is represented by a complex vector of size 2:

$$\begin{bmatrix} \alpha \\ \beta \end{bmatrix}$$

Here 𝛼 and 𝛽 are complex numbers. 𝛼 represents how "close" the qubit is to state 0, and 𝛽 represents how "close" the qubit is to state 1. 

This vector is normalized: $|\alpha|^2 + |\beta|^2 = 1$

> How to calculate the [[Complex Numbers#Modulus|Complex Modulus]] here.

𝛼 and 𝛽 are known as the probability amplitudes of states 0 and 1, respectively.

## Basis States
---

A qubit in state 0 would be represented by the following vector:

$$\begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

Likewise, a qubit in state 1 would be represented by this vector:

$$\begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

Note that you can use scalar multiplication and vector addition to express any qubit state $\begin{bmatrix} \alpha \\ \beta \end{bmatrix}$ as a sum of these two vectors with certain weights 𝛼 and 𝛽, known as linear combination.

$$\begin{bmatrix} \alpha \\ \beta \end{bmatrix} =
\begin{bmatrix} \alpha \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ \beta \end{bmatrix} =
\alpha \cdot \begin{bmatrix} 1 \\ 0 \end{bmatrix} + \beta \cdot \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

Because of this, qubit states 0 and 1 are known as basis states. These two vectors have two properties.

1. They are normalized.

$$\langle \begin{bmatrix} 1 \\ 0 \end{bmatrix} , \begin{bmatrix} 1 \\ 0 \end{bmatrix} \rangle =
 \langle \begin{bmatrix} 0 \\ 1 \end{bmatrix} , \begin{bmatrix} 0 \\ 1 \end{bmatrix} \rangle = 1$$

2. They are orthogonal to each other.

$$\langle \begin{bmatrix} 1 \\ 0 \end{bmatrix} , \begin{bmatrix} 0 \\ 1 \end{bmatrix} \rangle =
 \langle \begin{bmatrix} 0 \\ 1 \end{bmatrix} , \begin{bmatrix} 1 \\ 0 \end{bmatrix} \rangle = 0$$

> As a reminder, ⟨𝑉,𝑊⟩ is the inner product of 𝑉 and 𝑊.

This means that these vectors form an **orthonormal basis**. The basis of $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$ is called the computational basis, also known as the canonical basis.

>There exist other orthonormal bases, for example, the **Hadamard basis**, formed by the vectors
>$$\begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} \text{ and } \begin{bmatrix} \frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} \end{bmatrix}$$ 
   You can check that these vectors are normalized, and orthogonal to each other. Any qubit state  can be expressed as a linear combination of these vectors:
   >$$\begin{bmatrix} \alpha \\ \beta \end{bmatrix} =
\frac{\alpha + \beta}{\sqrt{2}} \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} +
\frac{\alpha - \beta}{\sqrt{2}} \begin{bmatrix} \frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} \end{bmatrix}$$
>The Hadamard basis is widely used in quantum computing, for example, in the [BB84 quantum key distribution protocol](https://en.wikipedia.org/wiki/BB84).


## The Dirac Notation
---

Dirac notation is a shorthand notation that eases writing quantum states and computing linear algebra. In Dirac notation, a vector is denoted by a symbol called a **ket**. For example, a qubit in state 0 is represented by the ket |0⟩, and a qubit in state 1 is represented by the ket |1⟩:

| $\ket{0} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ | $\ket{1} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ |
| ------------------------------------------------ | ------------------------------------------------ |

These two kets represent basis states, so they can be used to represent any other state:

$$\begin{bmatrix} \alpha \\ \beta \end{bmatrix} = \alpha\ket{0} + \beta\ket{1}$$

Dirac notation is not only restricted to vectors 0 and 1; it can be used to represent any vector, similar to how variable names are used in algebra. For example, we can call the state above "the state 𝜓" and write it as:

$$\ket{\psi} = \alpha\ket{0} + \beta\ket{1}$$

Several ket symbols have a generally accepted use, so you will see them often:

| $\ket{+} = \frac{1}{\sqrt{2}}\big(\ket{0} + \ket{1}\big)$  | $\ket{-} = \frac{1}{\sqrt{2}}\big(\ket{0} - \ket{1}\big)$   |
| ---------------------------------------------------------- | ----------------------------------------------------------- |
| $\ket{i} = \frac{1}{\sqrt{2}}\big(\ket{0} + i\ket{1}\big)$ | $\ket{-i} = \frac{1}{\sqrt{2}}\big(\ket{0} - i\ket{1}\big)$ |
We will learn more about Dirac notation in the next katas, as we introduce quantum gates and multi-qubit systems.

## Relative and Global Phase
---

Complex numbers have a parameter called the phase. If a complex number 𝑧=𝑥+𝑖𝑦 is written in polar form $z = re^{i\theta}$, its phase is 𝜃, where 𝜃=𝑎𝑡𝑎𝑛2(𝑦,𝑥).

>`atan2` is a useful function available in most programming languages. It takes two arguments and returns an angle 𝜃 between −𝜋 and 𝜋 that has cos⁡𝜃=𝑥 and sin⁡𝜃=𝑦. Unlike using $\tan^{-1}(\frac{y}{x})$, `atan2` computes the correct quadrant for the angle, since it preserves information about the signs of both sine and cosine of the angle.

The probability amplitudes 𝛼 and 𝛽 are complex numbers, therefore 𝛼 and 𝛽 have a phase. For example, consider a qubit in state $\frac{1 + i}{2}\ket{0} + \frac{1 - i}{2}\ket{1}$ If you do the math, you see that the phase of |0⟩ is $atan2(\frac12, \frac12) = \frac{\pi}{4}$. The difference between these two phases is known as **relative phase**.

Multiplying the state of the entire system by $e^{i\theta}$ doesn't affect the relative phase: 𝛼|0⟩+𝛽|1⟩ has the same relative phase as $e^{i\theta}\big(\alpha\ket{0} + \beta\ket{1}\big)$. In the second expression, 𝜃 is known as the system's **global phase**.

The state of a qubit (or, more generally, the state of a quantum system) is defined by its relative phase - global phase arises as a consequence of using linear algebra to represent qubits, and has no physical meaning. That is, applying a phase to the entire state of a system (multiplying the entire vector by $e^{i\theta}$ for any real 𝜃) doesn't actually affect the state of the system. Because of this, global phase is sometimes known as **unobservable phase** or **hidden phase**.

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

$$\begin{bmatrix} \alpha \\ \beta \end{bmatrix} = \alpha\ket{0} + \beta\ket{1}$$

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