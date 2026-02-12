#include <gtest/gtest.h>
#include "ProblemTypes/Common.h"

class CommonTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Инициализация общих данных для тестов
        A.resize(2, 2);
        A << 1, 2,
             3, 4;
        
        b.resize(2);
        b << 5, 6;
        
        c.resize(2);
        c << 7, 8;
        
        constraintTypes = {
            Common::ConstraintType::LessOrEqual,
            Common::ConstraintType::GreaterOrEqual
        };
        
        variableTypes = {
            Common::VariableType::NonNegative,
            Common::VariableType::NonNegative
        };
    }
    
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    std::vector<Common::ConstraintType> constraintTypes;
    std::vector<Common::VariableType> variableTypes;
};

TEST_F(CommonTest, CreationTest)
{
    Common common(A, b, c, constraintTypes, variableTypes, true);
    
    EXPECT_TRUE(common.IsMaximization());
    EXPECT_EQ(common.GetConstraintsMatrix().rows(), 2);
    EXPECT_EQ(common.GetConstraintsMatrix().cols(), 2);
    EXPECT_EQ(common.GetRightHandSide().size(), 2);
    EXPECT_EQ(common.GetObjectiveCoefficients().size(), 2);
}

TEST_F(CommonTest, EvaluateTest)
{
    Common common(A, b, c, constraintTypes, variableTypes, true);
    
    Eigen::VectorXd solution(2);
    solution << 1, 2;
    
    // c^T * x = 7*1 + 8*2 = 23
    double value = common.Evaluate(solution);
    EXPECT_DOUBLE_EQ(value, 23.0);
}

TEST_F(CommonTest, InvalidDimensionsTest)
{
    Eigen::VectorXd wrong_b(3);
    wrong_b << 1, 2, 3;
    
    EXPECT_THROW(
        Common(A, wrong_b, c, constraintTypes, variableTypes, true),
        std::invalid_argument
    );
}

TEST_F(CommonTest, CopyConstructorTest)
{
    Common common1(A, b, c, constraintTypes, variableTypes, true);
    Common common2(common1);
    
    EXPECT_TRUE(common2.IsMaximization());
    EXPECT_EQ(common2.GetConstraintsMatrix().rows(), 2);
}

TEST_F(CommonTest, GetDualTest)
{
    Common common(A, b, c, constraintTypes, variableTypes, true);
    
    auto dual = common.GetDual();
    ASSERT_NE(dual, nullptr);
    
    // Двойственная задача должна быть на минимизацию
    EXPECT_FALSE(dual->IsMaximization());
    
    // Матрица должна быть транспонирована
    EXPECT_EQ(dual->GetConstraintsMatrix().rows(), 2);
    EXPECT_EQ(dual->GetConstraintsMatrix().cols(), 2);
}
