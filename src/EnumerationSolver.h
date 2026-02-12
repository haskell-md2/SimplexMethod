#pragma once
#include "ProblemTypes/Canonical.h"

class EnumerationSolver {
private:

    const Canonical& problem;

public:

    double solveEnumeration();

    explicit EnumerationSolver(const Canonical &problem);
    ~EnumerationSolver();
};
