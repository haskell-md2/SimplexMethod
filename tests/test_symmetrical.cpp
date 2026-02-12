#include <gtest/gtest.h>
#include "ProblemTypes/Symmetrical.h"
#include "ProblemTypes/Common.h"
#include "ProblemTypes/Canonical.h"

class SymmetricalTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        A.resize(2, 2);
        A << 1, 2,
             3, 4;
        
        b.resize(2);
        b << 5, 6;
        
        c.resize(2);
        c << 7, 8;
    }
    
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
};

TEST_F(SymmetricalTest, CreationTest)
{
    Symmetrical sym(A, b, c, true);
    
    EXPECT_TRUE(sym.IsMaximization());
    EXPECT_EQ(sym.GetConstraintsMatrix().rows(), 2);
    EXPECT_EQ(sym.GetConstraintsMatrix().cols(), 2);
}

TEST_F(SymmetricalTest, GetDualTest)
{
    Symmetrical sym(A, b, c, true);
    
    auto dual = sym.GetDual();
    ASSERT_NE(dual, nullptr);
    
    // Двойственная задача на минимизацию
    EXPECT_FALSE(dual->IsMaximization());
    
    // Проверяем транспонирование
    EXPECT_EQ(dual->GetConstraintsMatrix().rows(), A.cols());
    EXPECT_EQ(dual->GetConstraintsMatrix().cols(), A.rows());
    
    // Проверяем обмен b и c
    EXPECT_TRUE(dual->GetRightHandSide().isApprox(c));
    EXPECT_TRUE(dual->GetObjectiveCoefficients().isApprox(b));
}

TEST_F(SymmetricalTest, ToCanonicalMaximizationTest)
{
    Symmetrical sym(A, b, c, true);
    
    auto canonical = sym.ToCanonical();
    ASSERT_NE(canonical, nullptr);
    
    // Для максимизации добавляются slack переменные
    // Новая размерность: n + m = 2 + 2 = 4
    EXPECT_EQ(canonical->GetConstraintsMatrix().cols(), 4);
    EXPECT_EQ(canonical->GetConstraintsMatrix().rows(), 2);
    
    // Базис состоит из slack переменных
    const auto& basis = canonical->GetBasisIndices();
    EXPECT_EQ(basis.size(), 2);
    EXPECT_EQ(basis[0], 2);  // Третья переменная (slack)
    EXPECT_EQ(basis[1], 3);  // Четвертая переменная (slack)
}

TEST_F(SymmetricalTest, ToCanonicalMinimizationTest)
{
    Symmetrical sym(A, b, c, false);
    
    auto canonical = sym.ToCanonical();
    ASSERT_NE(canonical, nullptr);
    
    // Для минимизации добавляются surplus и artificial
    // Новая размерность: n + 2m = 2 + 4 = 6
    EXPECT_EQ(canonical->GetConstraintsMatrix().cols(), 6);
    EXPECT_EQ(canonical->GetConstraintsMatrix().rows(), 2);
}

TEST_F(SymmetricalTest, ToCommonTest)
{
    Symmetrical sym(A, b, c, true);
    
    auto common = sym.ToCommon();
    ASSERT_NE(common, nullptr);
    
    EXPECT_TRUE(common->IsMaximization());
    EXPECT_EQ(common->GetConstraintTypes().size(), 2);
    EXPECT_EQ(common->GetVariableTypes().size(), 2);
}
