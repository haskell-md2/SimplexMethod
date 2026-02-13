#include <gtest/gtest.h>
#include "ProblemTypes/Canonical.h"
#include "ProblemTypes/Common.h"
#include "ProblemTypes/Canonical.h"

class CanonicalTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Создаем простую каноническую задачу со slack переменными
        A.resize(2, 4);
        A << 1, 2, 1, 0,
             3, 4, 0, 1;
        
        b.resize(2);
        b << 5, 6;
        
        c.resize(4);
        c << 7, 8, 0, 0;  // Slack переменные имеют коэффициент 0
        
        basisIndices = {2, 3};  // Slack переменные в базисе
    }
    
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    std::vector<int> basisIndices;
};

TEST_F(CanonicalTest, CreationTest)
{
    Canonical canonical(A, b, c, basisIndices, true);
    
    EXPECT_FALSE(canonical.IsMaximization());  // minimize = true
    EXPECT_EQ(canonical.GetConstraintsMatrix().rows(), 2);
    EXPECT_EQ(canonical.GetConstraintsMatrix().cols(), 4);
    EXPECT_EQ(canonical.GetBasisIndices().size(), 2);
}

TEST_F(CanonicalTest, GetBasicSolutionTest)
{
    Canonical canonical(A, b, c, basisIndices, true);
    canonical.SetOriginalVariablesCount(2);
    
    Eigen::VectorXd solution = canonical.GetBasicSolution();
    
    // Проверяем размерность
    EXPECT_EQ(solution.size(), 4);
    
    // Небазисные переменные должны быть 0
    EXPECT_DOUBLE_EQ(solution[0], 0.0);
    EXPECT_DOUBLE_EQ(solution[1], 0.0);
    
    // Базисные переменные (slack) должны равняться b
    EXPECT_DOUBLE_EQ(solution[2], 5.0);
    EXPECT_DOUBLE_EQ(solution[3], 6.0);
}

TEST_F(CanonicalTest, IsFeasibleBasisTest)
{
    Canonical canonical(A, b, c, basisIndices, true);
    
    // С положительным b базис должен быть допустимым
    EXPECT_TRUE(canonical.IsFeasibleBasis());
}

TEST_F(CanonicalTest, InvalidBasisIndicesTest)
{
    std::vector<int> invalidBasis = {10, 20};  // Выходят за пределы
    
    EXPECT_THROW(
        Canonical(A, b, c, invalidBasis, true),
        std::invalid_argument
    );
}

TEST_F(CanonicalTest, ToCommonTest)
{
    Canonical canonical(A, b, c, basisIndices, true);
    canonical.SetOriginalVariablesCount(2);
    
    auto common = canonical.ToCommon();
    ASSERT_NE(common, nullptr);
    
    // Должны остаться только исходные переменные
    EXPECT_EQ(common->GetObjectiveCoefficients().size(), 2);
    EXPECT_EQ(common->GetConstraintsMatrix().cols(), 2);
}
