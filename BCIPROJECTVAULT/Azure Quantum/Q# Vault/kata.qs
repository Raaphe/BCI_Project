namespace Kata {
    operation FlipQubit(q : Qubit) : Unit is Adj + Ctl {
        X(q);
    }
}