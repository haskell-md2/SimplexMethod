#pragma once

#include "IProblem.h"
#include <memory>
#include <vector>

class Symmetrical;
class Canonical;

class Common : public IProblem
{
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;   

public:
    enum class ConstraintType {
        LessOrEqual, 
        GreaterOrEqual,
        Equal
    };

    enum class VariableType {
        Free,
        NonNegative,
        NonPositive
    };

    Common(const Eigen::MatrixXd& A,
           const Eigen::VectorXd& b,
           const Eigen::VectorXd& c,
           const std::vector<ConstraintType>& constraintTypes,
           const std::vector<VariableType>& variableTypes,
           bool maximize);
    
    Common(const Common& other);
    Common(Common&& other) noexcept;
    Common& operator=(const Common& other); 
    Common& operator=(Common&& other) noexcept;
    ~Common() override;
    
    double Evaluate(const Eigen::VectorXd& solution) const override;
    void Print() const override;

    const Eigen::MatrixXd& GetConstraintsMatrix() const override;
    const Eigen::VectorXd& GetRightHandSide() const override;
    const Eigen::VectorXd& GetObjectiveCoefficients() const override;
    bool IsMaximization() const override;

    const std::vector<ConstraintType>& GetConstraintTypes() const;
    const std::vector<VariableType>& GetVariableTypes() const;

    std::unique_ptr<Symmetrical> ToSymmetrical() const;
    std::unique_ptr<Canonical> ToCanonical() const;
    std::unique_ptr<Common> GetDual() const;
};
