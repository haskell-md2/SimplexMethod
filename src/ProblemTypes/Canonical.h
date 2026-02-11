#pragma once

#include "IProblem.h"


class Canonical : public IProblem
{
private:
    Eigen::MatrixXd _A;
    Eigen::VectorXd _b;
    Eigen::VectorXd _c;

public:
    
    IProblem * GetDual() {return this;};
    void Print() {};

    Eigen::MatrixXd getA(){return _A;}
    Eigen::VectorXd getb() {return _b;}
    Eigen::VectorXd getc() {return _c;}

    Canonical(Eigen::MatrixXd A, Eigen::VectorXd b, Eigen::VectorXd c): _A(A), _b(b), _c(c) {};
    ~Canonical() {};
};


