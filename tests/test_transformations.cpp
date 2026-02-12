#include <gtest/gtest.h>
#include "ProblemTypes/Common.h"
#include "ProblemTypes/Symmetrical.h"
#include "ProblemTypes/Canonical.h"

TEST(TransformationsTest, CommonToSymmetricalToCanonical)
{
    Eigen::MatrixXd A(2, 2);
    A << 1, 2,
         3, 4;
    
    Eigen::VectorXd b(2);
    b << 5, 6;
    
    Eigen::VectorXd c(2);
    c << 7, 8;
    
    std::vector<Common::ConstraintType> constraints = {
        Common::ConstraintType::LessOrEqual,
        Common::ConstraintType::LessOrEqual
    };
    
    std::vector<Common::VariableType> variables = {
        Common::VariableType::NonNegative,
        Common::VariableType::NonNegative
    };
    
    // Common → Symmetrical → Canonical
    Common common(A, b, c, constraints, variables, true);
    auto sym = common.ToSymmetrical();
    ASSERT_NE(sym, nullptr);
    
    auto canonical = sym->ToCanonical();
    ASSERT_NE(canonical, nullptr);
    
    EXPECT_EQ(canonical->GetConstraintsMatrix().rows(), 2);
}

TEST(TransformationsTest, SymmetricalDualOfDual)
{
    Eigen::MatrixXd A(2, 2);
    A << 1, 2,
         3, 4;
    
    Eigen::VectorXd b(2);
    b << 5, 6;
    
    Eigen::VectorXd c(2);
    c << 7, 8;
    
    Symmetrical sym(A, b, c, true);
    
    // Двойственная от двойственной должна быть близка к исходной
    auto dual = sym.GetDual();
    auto dualOfDual = dual->GetDual();
    
    EXPECT_TRUE(dualOfDual->IsMaximization());
    EXPECT_TRUE(dualOfDual->GetConstraintsMatrix().isApprox(A));
    EXPECT_TRUE(dualOfDual->GetRightHandSide().isApprox(b));
    EXPECT_TRUE(dualOfDual->GetObjectiveCoefficients().isApprox(c));
}
