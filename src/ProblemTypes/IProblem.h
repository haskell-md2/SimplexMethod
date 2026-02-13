#pragma once

#include <Eigen/Dense>
#include <memory>
#include <string>

class IProblem
{
public:
    virtual double Evaluate(const Eigen::VectorXd& solution) const = 0;
    virtual void Print() const = 0;
    virtual const Eigen::MatrixXd& GetConstraintsMatrix() const = 0;
    virtual const Eigen::VectorXd& GetRightHandSide() const = 0;
    virtual const Eigen::VectorXd& GetObjectiveCoefficients() const = 0;
    virtual bool IsMaximization() const = 0;
    virtual ~IProblem() = default;
};