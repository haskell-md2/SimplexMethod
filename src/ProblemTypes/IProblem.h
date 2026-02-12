#pragma once

#include <Eigen/Dense>
#include <string>

class IProblem
{

public:
    virtual double Evaluate(const Eigen::VectorXd& solution) const = 0;

    virtual void Print() = 0;f

    virtual std::unique_ptr<IProblem> GetDual() const = 0;

    virtual const Eigen::MatrixXd& GetConstraintsMatrix() const = 0;
    virtual const Eigen::MatrixXd& GetRideHandSide() const = 0;
    virtual const Eigen::MatrixXd& GetObjectiveCoefficients() const = 0;
    virtual bool IsMaximization() const = 0;
    
    virtual ~IProblem() = default;
};

