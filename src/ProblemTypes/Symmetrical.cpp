#include "Symmetrical.h"
#include "Common.h"
#include "Canonical.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>

struct Symmetrical::Impl
{
    Eigen::MatrixXd A;  
    Eigen::VectorXd b;  
    Eigen::VectorXd c;  
    bool maximize;      

    Impl(const Eigen::MatrixXd& A_,
         const Eigen::VectorXd& b_,
         const Eigen::VectorXd& c_,
         bool maximize_)
        : A(A_), b(b_), c(c_), maximize(maximize_)
    {
        if (A.rows() != b.size())
        {
            throw std::invalid_argument("Размерность матрицы A и вектора b не совпадают");
        }
        if (A.cols() != c.size())
        {
            throw std::invalid_argument("Размерность матрицы A и вектора c не совпадают");
        }
    }
};


Symmetrical::Symmetrical(const Eigen::MatrixXd& A,
                         const Eigen::VectorXd& b,
                         const Eigen::VectorXd& c,
                         bool maximize)
    : pimpl_(std::make_unique<Impl>(A, b, c, maximize))
{
}

Symmetrical::Symmetrical(const Symmetrical& other)
    : pimpl_(std::make_unique<Impl>(*other.pimpl_))
{
}

Symmetrical::Symmetrical(Symmetrical&& other) noexcept = default;

Symmetrical& Symmetrical::operator=(const Symmetrical& other)
{
    if (this != &other)
    {
        pimpl_ = std::make_unique<Impl>(*other.pimpl_);
    }
    return *this;
}

Symmetrical& Symmetrical::operator=(Symmetrical&& other) noexcept = default;

Symmetrical::~Symmetrical() = default;

double Symmetrical::Evaluate(const Eigen::VectorXd& solution) const
{
    if (solution.size() != pimpl_->c.size())
    {
        throw std::invalid_argument("Размерность решения не совпадает с количеством переменных");
    }
    
    return pimpl_->c.dot(solution);
}

void Symmetrical::Print() const
{
    std::cout << "=== Симметричная форма задачи ЛП ===" << std::endl;
    std::cout << (pimpl_->maximize ? "Максимизировать: " : "Минимизировать: ");
    
    for (int i = 0; i < pimpl_->c.size(); ++i)
    {
        if (i > 0 && pimpl_->c[i] >= 0) std::cout << " + ";
        std::cout << pimpl_->c[i] << "*x" << (i + 1);
    }
    std::cout << std::endl << std::endl;

    std::cout << "При ограничениях:" << std::endl;
    const char* inequality = pimpl_->maximize ? " <= " : " >= ";
    
    for (int i = 0; i < pimpl_->A.rows(); ++i)
    {
        for (int j = 0; j < pimpl_->A.cols(); ++j)
        {
            if (j > 0 && pimpl_->A(i, j) >= 0) std::cout << " + ";
            std::cout << pimpl_->A(i, j) << "*x" << (j + 1);
        }
        std::cout << inequality << pimpl_->b[i] << std::endl;
    }

    std::cout << std::endl << "Все переменные неотрицательны: x_i >= 0" << std::endl;
}

const Eigen::MatrixXd& Symmetrical::GetConstraintsMatrix() const
{
    return pimpl_->A;
}

const Eigen::VectorXd& Symmetrical::GetRightHandSide() const
{
    return pimpl_->b;
}

const Eigen::VectorXd& Symmetrical::GetObjectiveCoefficients() const
{
    return pimpl_->c;
}

bool Symmetrical::IsMaximization() const
{
    return pimpl_->maximize;
}

std::unique_ptr<Symmetrical> Symmetrical::GetDual() const
{
    // Построение двойственной задачи
    // Прямая задача (max):     Двойственная задача (min):
    // max c^T x                min b^T y
    // Ax ≤ b                   A^T y ≥ c
    // x ≥ 0                    y ≥ 0
    //
    // Если прямая на min:      Двойственная на max:
    // min c^T x                max b^T y
    // Ax ≥ b                   A^T y ≤ c
    // x ≥ 0                    y ≥ 0

    Eigen::MatrixXd A_dual = pimpl_->A.transpose();
    
    Eigen::VectorXd b_dual = pimpl_->c;
    Eigen::VectorXd c_dual = pimpl_->b;
    
    bool maximize_dual = !pimpl_->maximize;
    
    return std::make_unique<Symmetrical>(A_dual, b_dual, c_dual, maximize_dual);
}

std::unique_ptr<Canonical> Symmetrical::ToCanonical() const
{
    // TODO: Реализация преобразования в каноническую форму
    // Алгоритм:
    // 1. Если максимизация (Ax ≤ b):
    //    - Добавляем дополнительные переменные (slack): Ax + s = b, s ≥ 0
    //    - Базис: s_1, s_2, ..., s_m
    //    - Расширяем c нулями для slack переменных
    //
    // 2. Если минимизация (Ax ≥ b):
    //    - Вычитаем избыточные переменные (surplus): Ax - s = b, s ≥ 0
    //    - Проблема: нет очевидного базиса
    //    - Добавляем искусственные переменные: Ax - s + a = b, a ≥ 0
    //    - Базис: a_1, a_2, ..., a_m
    //    - Метод искусственного базиса (Big M или двухфазный симплекс)
    //
    // 3. Преобразуем минимизацию в максимизацию (min c^T x = -max (-c)^T x)

    int m = pimpl_->A.rows();  // Количество ограничений
    int n = pimpl_->A.cols();  // Количество исходных переменных

    if (pimpl_->maximize)
    {
        // Случай максимизации: Ax ≤ b
        // Добавляем slack переменные
        
        // Новая матрица A: [A | I]
        Eigen::MatrixXd A_canonical(m, n + m);
        A_canonical.leftCols(n) = pimpl_->A;
        A_canonical.rightCols(m) = Eigen::MatrixXd::Identity(m, m);
        
        // Новый вектор c: [c | 0]
        Eigen::VectorXd c_canonical = Eigen::VectorXd::Zero(n + m);
        c_canonical.head(n) = pimpl_->c;
        
        // Базисные индексы: slack переменные (индексы n, n+1, ..., n+m-1)
        std::vector<int> basisIndices(m);
        for (int i = 0; i < m; ++i)
        {
            basisIndices[i] = n + i;
        }
        
        // Создаем каноническую форму (максимизация)
        auto canonical = std::make_unique<Canonical>(A_canonical, pimpl_->b, c_canonical, basisIndices, false);
        canonical->SetOriginalVariablesCount(n);
        
        return canonical;
    }
    else
    {
        // Случай минимизации: Ax ≥ b
        // Добавляем surplus и artificial переменные
        
        // Новая матрица A: [A | -I | I]
        // A - исходная матрица
        // -I - surplus переменные
        // I - artificial переменные
        Eigen::MatrixXd A_canonical(m, n + 2 * m);
        A_canonical.leftCols(n) = pimpl_->A;
        A_canonical.block(0, n, m, m) = -Eigen::MatrixXd::Identity(m, m);  // surplus
        A_canonical.rightCols(m) = Eigen::MatrixXd::Identity(m, m);         // artificial
        
        // Новый вектор c для минимизации: [c | 0 | 0]
        Eigen::VectorXd c_canonical = Eigen::VectorXd::Zero(n + 2 * m);
        c_canonical.head(n) = pimpl_->c;
        // surplus и artificial переменные имеют коэффициент 0 в целевой функции
        // (Big M метод будет реализован в SimplexSolver)
        
        // Базисные индексы: artificial переменные (индексы n+m, n+m+1, ..., n+2m-1)
        std::vector<int> basisIndices(m);
        for (int i = 0; i < m; ++i)
        {
            basisIndices[i] = n + m + i;
        }
        
        // Создаем каноническую форму (минимизация)
        auto canonical = std::make_unique<Canonical>(A_canonical, pimpl_->b, c_canonical, basisIndices, true);
        canonical->SetOriginalVariablesCount(n);
        
        return canonical;
    }
}

std::unique_ptr<Common> Symmetrical::ToCommon() const
{
    // Преобразование симметричной формы в общую форму
    // Симметричная форма:
    // max c^T x (или min)
    // Ax <= b (для максимизации) или Ax >= b (для минимизации)
    // x ≥ 0
    //
    // Общая форма:
    // - Все ограничения одного типа (соответствуют типу из симметричной)
    // - Все переменные типа NonNegative (≥ 0)

    int m = pimpl_->A.rows();
    int n = pimpl_->A.cols();

    // Матрицы и векторы переносятся без изменений
    Eigen::MatrixXd A_common = pimpl_->A;
    Eigen::VectorXd b_common = pimpl_->b;
    Eigen::VectorXd c_common = pimpl_->c;

    // Определяем типы ограничений в зависимости от типа оптимизации
    std::vector<Common::ConstraintType> constraintTypes(m);
    
    if (pimpl_->maximize)
    {
        // Для максимизации: все ограничения типа <=
        for (int i = 0; i < m; ++i)
        {
            constraintTypes[i] = Common::ConstraintType::LessOrEqual;
        }
    }
    else
    {
        // Для минимизации: все ограничения типа >=
        for (int i = 0; i < m; ++i)
        {
            constraintTypes[i] = Common::ConstraintType::GreaterOrEqual;
        }
    }

    // Все переменные типа NonNegative
    std::vector<Common::VariableType> variableTypes(n, Common::VariableType::NonNegative);

    // Тип оптимизации остается тем же
    bool maximize_common = pimpl_->maximize;

    return std::make_unique<Common>(A_common, b_common, c_common, 
                                    constraintTypes, variableTypes, maximize_common);
}
