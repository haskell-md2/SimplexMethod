#pragma once
#include "ProblemTypes/Canonical.h"

class EnumerationSolver {

    const double tol_rank = 1e-12;
    const double tol_feas = 1e-9;
    const double tol_res  = 1e-8;
    const double tol_dup  = 1e-9;


    const Canonical& problem;

    std::vector<Eigen::VectorXd> found_vertices;

public:

    void solveEnumeration();

    explicit EnumerationSolver(const Canonical &problem);
    ~EnumerationSolver();

    bool handleSolution(const Eigen::VectorXd &xB, int rank, int A_num_cols, const std::vector<int> &cols);

    double findSolution();
};
