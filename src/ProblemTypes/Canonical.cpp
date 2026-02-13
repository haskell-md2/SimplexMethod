#include "Canonical.h"
#include "Common.h"
#include "Symmetrical.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>

struct Canonical::Impl
{
    Eigen::MatrixXd A;              
    Eigen::VectorXd b;              
    Eigen::VectorXd c;              
    std::vector<int> basisIndices;  
    bool minimize;                  
    int originalVariablesCount;     

    Impl(const Eigen::MatrixXd& A_,
         const Eigen::VectorXd& b_,
         const Eigen::VectorXd& c_,
         const std::vector<int>& basisIndices_,
         bool minimize_)
        : A(A_), b(b_), c(c_),
          basisIndices(basisIndices_),
          minimize(minimize_),
          originalVariablesCount(c_.size())
    {
        if (A.rows() != b.size())
        {
            throw std::invalid_argument("Размерность матрицы A и вектора b не совпадают");
        }
        if (A.cols() != c.size())
        {
            throw std::invalid_argument("Размерность матрицы A и вектора c не совпадают");
        }
        if (static_cast<int>(basisIndices.size()) != A.rows())
        {
            throw std::invalid_argument("Количество базисных индексов не совпадает с количеством строк A");
        }

        for (int idx : basisIndices)
        {
            if (idx < 0 || idx >= A.cols())
            {
                throw std::invalid_argument("Базисный индекс выходит за пределы допустимого диапазона");
            }
        }
    }
};

Canonical::Canonical(const Eigen::MatrixXd& A,
                     const Eigen::VectorXd& b,
                     const Eigen::VectorXd& c,
                     const std::vector<int>& basisIndices,
                     bool minimize)
    : pimpl_(std::make_unique<Impl>(A, b, c, basisIndices, minimize))
{
}

Canonical::Canonical(const Canonical& other)
    : pimpl_(std::make_unique<Impl>(*other.pimpl_))
{
}

Canonical::Canonical(Canonical&& other) noexcept = default;

Canonical& Canonical::operator=(const Canonical& other)
{
    if (this != &other)
    {
        pimpl_ = std::make_unique<Impl>(*other.pimpl_);
    }
    return *this;
}

Canonical& Canonical::operator=(Canonical&& other) noexcept = default;

Canonical::~Canonical() = default;

double Canonical::Evaluate(const Eigen::VectorXd& solution) const
{
    if (solution.size() != pimpl_->c.size())
    {
        throw std::invalid_argument("Размерность решения не совпадает с количеством переменных");
    }
    
    return pimpl_->c.dot(solution);
}

void Canonical::Print() const
{
    std::cout << "=== Каноническая форма задачи ЛП ===" << std::endl;
    std::cout << (pimpl_->minimize ? "Минимизировать: " : "Максимизировать: ");
    
    for (int i = 0; i < pimpl_->c.size(); ++i)
    {
        if (i > 0 && pimpl_->c[i] >= 0) std::cout << " + ";
        std::cout << pimpl_->c[i] << "*x" << (i + 1);
    }
    std::cout << std::endl << std::endl;

    std::cout << "При ограничениях (Ax = b):" << std::endl;
    for (int i = 0; i < pimpl_->A.rows(); ++i)
    {
        for (int j = 0; j < pimpl_->A.cols(); ++j)
        {
            if (j > 0 && pimpl_->A(i, j) >= 0) std::cout << " + ";
            std::cout << pimpl_->A(i, j) << "*x" << (j + 1);
        }
        std::cout << " = " << pimpl_->b[i] << std::endl;
    }

    std::cout << std::endl << "Все переменные неотрицательны: x_i >= 0" << std::endl;

    std::cout << std::endl << "Базисные переменные: ";
    for (size_t i = 0; i < pimpl_->basisIndices.size(); ++i)
    {
        if (i > 0) std::cout << ", ";
        std::cout << "x" << (pimpl_->basisIndices[i] + 1);
    }
    std::cout << std::endl;

    std::cout << "Количество исходных переменных: " << pimpl_->originalVariablesCount << std::endl;
    std::cout << "Дополнительных переменных: " << (pimpl_->c.size() - pimpl_->originalVariablesCount) << std::endl;
}

const Eigen::MatrixXd& Canonical::GetConstraintsMatrix() const
{
    return pimpl_->A;
}

const Eigen::VectorXd& Canonical::GetRightHandSide() const
{
    return pimpl_->b;
}

const Eigen::VectorXd& Canonical::GetObjectiveCoefficients() const
{
    return pimpl_->c;
}

bool Canonical::IsMaximization() const
{
    return !pimpl_->minimize;
}

const std::vector<int>& Canonical::GetBasisIndices() const
{
    return pimpl_->basisIndices;
}

int Canonical::GetOriginalVariablesCount() const
{
    return pimpl_->originalVariablesCount;
}

void Canonical::SetOriginalVariablesCount(int count)
{
    if (count <= 0 || count > pimpl_->c.size())
    {
        throw std::invalid_argument("Некорректное количество исходных переменных");
    }
    pimpl_->originalVariablesCount = count;
}

bool Canonical::IsFeasibleBasis() const
{
    Eigen::VectorXd solution = GetBasicSolution();
    
    for (int i = 0; i < solution.size(); ++i)
    {
        if (solution[i] < -1e-9)
        {
            return false;
        }
    }
    return true;
}

Eigen::VectorXd Canonical::GetBasicSolution() const
{
    Eigen::VectorXd solution = Eigen::VectorXd::Zero(pimpl_->c.size());
    
    Eigen::MatrixXd B(pimpl_->A.rows(), pimpl_->basisIndices.size());
    for (size_t i = 0; i < pimpl_->basisIndices.size(); ++i)
    {
        B.col(i) = pimpl_->A.col(pimpl_->basisIndices[i]);
    }
    
    Eigen::VectorXd basicValues = B.colPivHouseholderQr().solve(pimpl_->b);
    
    for (size_t i = 0; i < pimpl_->basisIndices.size(); ++i)
    {
        solution[pimpl_->basisIndices[i]] = basicValues[i];
    }
    
    return solution;
}

std::unique_ptr<Common> Canonical::ToCommon() const
{
    // Преобразование канонической формы в общую форму
    // Каноническая форма:
    // min/max c^T x
    // Ax = b
    // x ≥ 0
    //
    // Общая форма:
    // - Все ограничения типа Equal (=)
    // - Все переменные типа NonNegative (≥ 0)

    int m = pimpl_->A.rows();
    int n = pimpl_->originalVariablesCount;  // Только исходные переменные, без дополнительных

    // Извлекаем только исходные переменные (убираем slack/surplus/artificial)
    Eigen::MatrixXd A_common = pimpl_->A.leftCols(n);
    Eigen::VectorXd c_common = pimpl_->c.head(n);
    Eigen::VectorXd b_common = pimpl_->b;

    // Все ограничения типа Equal
    std::vector<Common::ConstraintType> constraintTypes(m, Common::ConstraintType::Equal);

    // Все переменные типа NonNegative
    std::vector<Common::VariableType> variableTypes(n, Common::VariableType::NonNegative);

    // Тип оптимизации
    bool maximize_common = !pimpl_->minimize;

    return std::make_unique<Common>(A_common, b_common, c_common, 
                                    constraintTypes, variableTypes, maximize_common);
}

std::unique_ptr<Symmetrical> Canonical::ToSymmetrical() const
{
    // Преобразование канонической формы в симметричную форму
    // Каноническая форма:
    // min/max c^T x
    // Ax = b
    // x ≥ 0
    //
    // Симметричная форма:
    // max c^T x (или min)
    // Ax <= b (для максимизации) или Ax >= b (для минимизации)
    // x ≥ 0
    //
    // Алгоритм:
    // Каждое равенство Ax = b заменяем на два неравенства:
    // - Ax <= b и Ax >= b (для максимизации)
    // - Затем Ax >= b превращаем в -Ax <= -b
    // Итого: каждое равенство → два неравенства <=
    //
    // Для минимизации аналогично, но с >=

    int m = pimpl_->A.rows();
    int n = pimpl_->originalVariablesCount;  // Только исходные переменные

    // Извлекаем только исходные переменные (убираем дополнительные)
    Eigen::MatrixXd A_original = pimpl_->A.leftCols(n);
    Eigen::VectorXd c_original = pimpl_->c.head(n);

    // Каждое равенство превращается в два неравенства
    int new_constraint_count = 2 * m;

    Eigen::MatrixXd A_sym(new_constraint_count, n);
    Eigen::VectorXd b_sym(new_constraint_count);

    if (!pimpl_->minimize)  // Максимизация
    {
        // Для максимизации: нужны неравенства <=
        // Ax = b → Ax <= b и -Ax <= -b
        for (int i = 0; i < m; ++i)
        {
            // Первое неравенство: Ax <= b
            A_sym.row(2 * i) = A_original.row(i);
            b_sym[2 * i] = pimpl_->b[i];

            // Второе неравенство: -Ax <= -b (из Ax >= b)
            A_sym.row(2 * i + 1) = -A_original.row(i);
            b_sym[2 * i + 1] = -pimpl_->b[i];
        }

        return std::make_unique<Symmetrical>(A_sym, b_sym, c_original, true);
    }
    else  // Минимизация
    {
        // Для минимизации: нужны неравенства >=
        // Ax = b → Ax >= b и -Ax >= -b
        // Но симметричная форма для минимизации использует >=
        // Поэтому оставляем как есть (не инвертируем)
        for (int i = 0; i < m; ++i)
        {
            // Первое неравенство: Ax >= b
            A_sym.row(2 * i) = A_original.row(i);
            b_sym[2 * i] = pimpl_->b[i];

            // Второе неравенство: -Ax >= -b (из Ax <= b)
            A_sym.row(2 * i + 1) = -A_original.row(i);
            b_sym[2 * i + 1] = -pimpl_->b[i];
        }

        return std::make_unique<Symmetrical>(A_sym, b_sym, c_original, false);
    }
}


std::unique_ptr<Canonical> Canonical::GetDual() const
{
    // Построение несимметричной двойственной задачи для канонической формы
    // Прямая задача (каноническая):
    // min c^T x
    // Ax = b
    // x ≥ 0
    //
    // Двойственная задача:
    // max b^T y
    // A^T y ≤ c
    // y — без ограничения на знак
    //
    // Для приведения к канонической форме нужно:
    // 1. Каждое неравенство A^T y ≤ c превратить в равенство добавлением slack переменных
    // 2. Каждую свободную переменную y_i заменить на разность y_i' - y_i''

    int m = pimpl_->A.rows();  // Количество ограничений прямой = количество переменных двойственной
    int n = pimpl_->A.cols();  // Количество переменных прямой = количество ограничений двойственной

    // Транспонируем матрицу A
    Eigen::MatrixXd A_T = pimpl_->A.transpose();
    
    // Каждую свободную переменную y_i заменяем на y_i' - y_i''
    // Получаем 2m переменных вместо m
    // Затем добавляем n slack переменных для неравенств A^T y ≤ c
    // Итого: 2m + n переменных
    
    Eigen::MatrixXd A_dual(n, 2 * m + n);
    
    // Заполняем первые 2m столбцов: [A^T | -A^T]
    A_dual.leftCols(m) = A_T;
    A_dual.middleCols(m, m) = -A_T;
    
    // Добавляем единичную матрицу для slack переменных
    A_dual.rightCols(n) = Eigen::MatrixXd::Identity(n, n);
    
    // Вектор правых частей: c
    Eigen::VectorXd b_dual = pimpl_->c;
    
    // Вектор коэффициентов целевой функции: [b | -b | 0]
    Eigen::VectorXd c_dual = Eigen::VectorXd::Zero(2 * m + n);
    c_dual.head(m) = pimpl_->b;
    c_dual.segment(m, m) = -pimpl_->b;
    
    // Базисные индексы: slack переменные (последние n переменных)
    std::vector<int> basisIndices_dual(n);
    for (int i = 0; i < n; ++i)
    {
        basisIndices_dual[i] = 2 * m + i;
    }
    
    // Двойственная задача на максимизацию (если прямая на минимизацию)
    bool minimize_dual = !pimpl_->minimize;
    
    auto canonical_dual = std::make_unique<Canonical>(A_dual, b_dual, c_dual, basisIndices_dual, minimize_dual);
    canonical_dual->SetOriginalVariablesCount(2 * m);  // Без slack переменных
    
    return canonical_dual;
}
