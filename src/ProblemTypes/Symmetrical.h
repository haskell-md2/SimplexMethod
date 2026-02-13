#pragma once

#include "IProblem.h"
#include <memory>

class Common;
class Canonical;

/**
 * @brief Симметричная (стандартная) форма задачи линейного программирования
 * Промежуточная форма между общей и канонической:
 * - Все ограничения одного типа (обычно ≤ для максимизации или ≥ для минимизации)
 * - Все переменные неотрицательные (x ≥ 0)
 * - Удобна для построения двойственной задачи
 */
class Symmetrical : public IProblem
{
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;

public:
    Symmetrical(const Eigen::MatrixXd& A,
                const Eigen::VectorXd& b,
                const Eigen::VectorXd& c,
                bool maximize);

    Symmetrical(const Symmetrical& other);
    Symmetrical(Symmetrical&& other) noexcept;
    Symmetrical& operator=(const Symmetrical& other);
    Symmetrical& operator=(Symmetrical&& other) noexcept;

    ~Symmetrical() override;

    double Evaluate(const Eigen::VectorXd& solution) const override;
    void Print() const override;
    const Eigen::MatrixXd& GetConstraintsMatrix() const override;
    const Eigen::VectorXd& GetRightHandSide() const override;
    const Eigen::VectorXd& GetObjectiveCoefficients() const override;
    bool IsMaximization() const override;

    std::unique_ptr<Symmetrical> GetDual() const;
    std::unique_ptr<Canonical> ToCanonical() const;
    std::unique_ptr<Common> ToCommon() const;
};
