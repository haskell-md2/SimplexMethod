#pragma once

#include "IProblem.h"
#include <memory>
#include <vector>

class Common;
class Symmetrical;

class Canonical : public IProblem
{
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;

public:

    void PrintCoffMatrixWithB();
    void PrintObjectiveCoefficients();
    Canonical(const Eigen::MatrixXd& A,
              const Eigen::VectorXd& b,
              const Eigen::VectorXd& c,
              const std::vector<int>& basisIndices,
              bool minimize = true);

    Canonical(const Canonical& other);
    Canonical(Canonical&& other) noexcept;
    Canonical& operator=(const Canonical& other);
    Canonical& operator=(Canonical&& other) noexcept;

    ~Canonical() override;

    double Evaluate(const Eigen::VectorXd& solution) const override;
    void Print() const override;
    const Eigen::MatrixXd& GetConstraintsMatrix() const override;
    const Eigen::VectorXd& GetRightHandSide() const override;
    const Eigen::VectorXd& GetObjectiveCoefficients() const override;
    bool IsMaximization() const override;

    const std::vector<int>& GetBasisIndices() const;

    int GetOriginalVariablesCount() const;

    void SetOriginalVariablesCount(int count);
    bool IsFeasibleBasis() const;
    Eigen::VectorXd GetBasicSolution() const;

    std::unique_ptr<Common> ToCommon() const;
    std::unique_ptr<Symmetrical> ToSymmetrical() const;
    std::unique_ptr<Canonical> GetDual() const;
};
