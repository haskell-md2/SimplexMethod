#pragma once

#include "IProblem.h"


class Canonical : public IProblem
{
public:
    Eigen::MatrixXd _A;
    Eigen::VectorXd _b;
    Eigen::VectorXd _c;

public:
    IProblem* GetDual() override;
    void Print() override;

    Canonical(Eigen::MatrixXd A, Eigen::VectorXd b, Eigen::VectorXd c): _A(A), _b(b), _c(c) {};
    ~Canonical() override;
};




