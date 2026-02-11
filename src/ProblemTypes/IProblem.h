#pragma once

#include <Eigen/Dense>

class IProblem
{

public:

    virtual IProblem * GetDual() = 0;
    virtual void Print() = 0;

    virtual ~IProblem() = default;
};

