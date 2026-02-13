#include "Common.h"
#include "Symmetrical.h"
#include "Canonical.h" 
#include "iostream"
#include "iomanip"
#include "stdexcept"


struct Common::Impl
{
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    std::vector<Common::ConstraintType> constraintTypes;
    std::vector<Common::VariableType> variableTypes;
    bool maximize;

    Impl(const Eigen::MatrixXd& A_,
         const Eigen::VectorXd& b_,
         const Eigen::VectorXd& c_,
         const std::vector<Common::ConstraintType>& constraintTypes_,
         const std::vector<Common::VariableType>& variableTypes_,
         bool maximize_)
        : A(A_), b(b_), c(c_),
          constraintTypes(constraintTypes_),
          variableTypes(variableTypes_),
          maximize(maximize_)
    {
        if (A.rows() != b.size())
        {
            throw std::invalid_argument("Размерность матрицы A и вектора b не совпадают");
        }
        if (A.cols() != c.size())
        {
            throw std::invalid_argument("Размерность матрицы A и вектора c не совпадают");
        }
        if (static_cast<int>(constraintTypes.size()) != A.rows())
        {
            throw std::invalid_argument("Количество типов ограничений не совпадает с количеством строк A");
        }
        if (static_cast<int>(variableTypes.size()) != A.cols())
        {
            throw std::invalid_argument("Количество типов переменных не совпадает с количеством столбцов A");
        }
    }
};
    
Common::Common(const Eigen::MatrixXd& A, 
               const Eigen::VectorXd& b, 
               const Eigen::VectorXd& c, 
               const std::vector<ConstraintType>& constraintTypes,  
               const std::vector<VariableType>& variableTypes,
                bool maximize)
    : pimpl_(std::make_unique<Impl>(A, b, c, constraintTypes, variableTypes, maximize))
{
}

Common::Common(const Common & other)
    : pimpl_(std::make_unique<Impl>(*other.pimpl_))
{
}

Common::Common(Common && other) noexcept = default;

Common& Common::operator=(const Common & other)
{
    if (this != &other)
    {
        pimpl_=std::make_unique<Impl>(*other.pimpl_);
    }
    return *this;
}

Common& Common::operator=(Common&& other) noexcept = default;

Common::~Common() = default;

double Common::Evaluate(const Eigen::VectorXd& solution) const {
    if (solution.size() != pimpl_->c.size()) {
        throw std::invalid_argument("Размерность решения не совпадает с количеством переменных");
    }

    return pimpl_->c.dot(solution);
}

void Common::Print() const {
    std::cout << "=== Общая форма задачи ЛП ===" << std::endl;
    std::cout << (pimpl_->maximize ? "Максимизировать: " : "Минимизировать: ");

    for (int i = 0; i < pimpl_->c.size(); ++i) {
        if (i > 0 && pimpl_->c[i] >= 0) std::cout << " + ";  
        std::cout << pimpl_->c[i] << "*x" << (i+1);  
    }

    std::cout << std::endl << std::endl;

    std::cout << "При ограничениях:" << std::endl;
    for (int i = 0; i < pimpl_->A.rows(); i++) {
        for (int j = 0; j <pimpl_->A.cols(); j++) {
            if (j > 0 && pimpl_->A(i,j) >= 0) std::cout << " + ";
            std::cout << pimpl_->A(i,j) << "x" << (j+1);
        }

        switch (pimpl_->constraintTypes[i])
        {
            case ConstraintType::LessOrEqual:
                std::cout << "<=";
                break;
            case ConstraintType::GreaterOrEqual:
                std::cout << ">=";
                break;
            case ConstraintType::Equal:
                std::cout << "=";
                break;
        }

        std::cout << pimpl_->b[i] << std::endl;
    }

    std::cout << std::endl << "Ограничения на переменные:" << std::endl;
    for (size_t i = 0; i < pimpl_->variableTypes.size(); i++) {
        std::cout << "x" << (i+1) << ": ";
        switch (pimpl_->variableTypes[i])
        {
            case VariableType::Free:
                std::cout << "∈R";
                break;
            case VariableType::NonNegative:
                std::cout << " >= 0";
                break;
            case VariableType::NonPositive:
                std::cout << " <= 0";
                break;
        }
        std::cout << std::endl;
    }
}

const Eigen::MatrixXd& Common::GetConstraintsMatrix() const
{
    return pimpl_->A;
}

const Eigen::VectorXd& Common::GetRightHandSide() const
{
    return pimpl_->b;
}

const Eigen::VectorXd& Common::GetObjectiveCoefficients() const
{
    return pimpl_->c;
}

bool Common::IsMaximization() const
{
    return pimpl_->maximize;
}

const std::vector<Common::ConstraintType>& Common::GetConstraintTypes() const
{
    return pimpl_->constraintTypes;
}

const std::vector<Common::VariableType>& Common::GetVariableTypes() const
{
    return pimpl_->variableTypes;
}

std::unique_ptr<Symmetrical> Common::ToSymmetrical() const
{
    // Алгоритм приведения к симметричной форме:
    // 1. Обработка переменных:
    //    - Свободные переменные x_j заменяем на разность: x_j = x_j' - x_j'', где x_j', x_j'' >= 0
    //    - Неположительные переменные x_j <= 0 заменяем на: x_j = -x_j', где x_j' >= 0
    //    - Неотрицательные переменные остаются без изменений
    // 2. Обработка ограничений:
    //    - Если максимизация:
    //      * Ограничения <= остаются без изменений
    //      * Ограничения >= умножаем на -1, получаем <=
    //      * Ограничения = заменяем на два: <= и >=, затем >= превращаем в <=
    //    - Если минимизация:
    //      * Ограничения >= остаются без изменений
    //      * Ограничения <= умножаем на -1, получаем >=
    //      * Ограничения = заменяем на два: >= и <=, затем <= превращаем в >=
    // 3. Если минимизация, умножаем целевую функцию на -1 и делаем максимизацию

    int m = pimpl_->A.rows();  // Количество ограничений
    int n = pimpl_->A.cols();  // Количество исходных переменных

    // Подсчитываем количество новых переменных
    int new_var_count = 0;
    for (int j = 0; j < n; ++j)
    {
        switch (pimpl_->variableTypes[j])
        {
            case VariableType::Free:
                new_var_count += 2;  // x_j' и x_j''
                break;
            case VariableType::NonNegative:
            case VariableType::NonPositive:
                new_var_count += 1;
                break;
        }
    }

    // Подсчитываем количество новых ограничений
    int new_constraint_count = 0;
    for (int i = 0; i < m; ++i)
    {
        if (pimpl_->constraintTypes[i] == ConstraintType::Equal)
        {
            new_constraint_count += 2;  // Равенство превращается в два неравенства
        }
        else
        {
            new_constraint_count += 1;
        }
    }

    // Создаем новую матрицу A
    Eigen::MatrixXd A_sym(new_constraint_count, new_var_count);
    Eigen::VectorXd b_sym(new_constraint_count);
    Eigen::VectorXd c_sym(new_var_count);

    // Заполняем новую матрицу A и вектор c
    int col_idx = 0;  // Индекс столбца в новой матрице
    for (int j = 0; j < n; ++j)
    {
        switch (pimpl_->variableTypes[j])
        {
            case VariableType::NonNegative:
            {
                // Переменная >= 0 остается без изменений
                c_sym[col_idx] = pimpl_->c[j];
                
                int row_idx = 0;
                for (int i = 0; i < m; ++i)
                {
                    if (pimpl_->constraintTypes[i] == ConstraintType::Equal)
                    {
                        // Для равенства добавляем в оба неравенства
                        A_sym(row_idx, col_idx) = pimpl_->A(i, j);
                        A_sym(row_idx + 1, col_idx) = -pimpl_->A(i, j);  // Для второго неравенства
                        row_idx += 2;
                    }
                    else if (pimpl_->constraintTypes[i] == ConstraintType::GreaterOrEqual)
                    {
                        // >= умножаем на -1 для получения <=
                        A_sym(row_idx, col_idx) = -pimpl_->A(i, j);
                        row_idx++;
                    }
                    else  // LessOrEqual
                    {
                        A_sym(row_idx, col_idx) = pimpl_->A(i, j);
                        row_idx++;
                    }
                }
                col_idx++;
                break;
            }

            case VariableType::NonPositive:
            {
                // x_j <= 0 заменяем на x_j = -x_j', где x_j' >= 0
                c_sym[col_idx] = -pimpl_->c[j];  // Знак меняется
                
                int row_idx = 0;
                for (int i = 0; i < m; ++i)
                {
                    if (pimpl_->constraintTypes[i] == ConstraintType::Equal)
                    {
                        A_sym(row_idx, col_idx) = -pimpl_->A(i, j);      // Знак меняется
                        A_sym(row_idx + 1, col_idx) = pimpl_->A(i, j);
                        row_idx += 2;
                    }
                    else if (pimpl_->constraintTypes[i] == ConstraintType::GreaterOrEqual)
                    {
                        A_sym(row_idx, col_idx) = pimpl_->A(i, j);  // -(-A) = A
                        row_idx++;
                    }
                    else  // LessOrEqual
                    {
                        A_sym(row_idx, col_idx) = -pimpl_->A(i, j);
                        row_idx++;
                    }
                }
                col_idx++;
                break;
            }

            case VariableType::Free:
            {
                // Свободная переменная x_j = x_j' - x_j''
                c_sym[col_idx] = pimpl_->c[j];      // Коэффициент для x_j'
                c_sym[col_idx + 1] = -pimpl_->c[j]; // Коэффициент для x_j''
                
                int row_idx = 0;
                for (int i = 0; i < m; ++i)
                {
                    if (pimpl_->constraintTypes[i] == ConstraintType::Equal)
                    {
                        A_sym(row_idx, col_idx) = pimpl_->A(i, j);
                        A_sym(row_idx, col_idx + 1) = -pimpl_->A(i, j);
                        A_sym(row_idx + 1, col_idx) = -pimpl_->A(i, j);
                        A_sym(row_idx + 1, col_idx + 1) = pimpl_->A(i, j);
                        row_idx += 2;
                    }
                    else if (pimpl_->constraintTypes[i] == ConstraintType::GreaterOrEqual)
                    {
                        A_sym(row_idx, col_idx) = -pimpl_->A(i, j);
                        A_sym(row_idx, col_idx + 1) = pimpl_->A(i, j);
                        row_idx++;
                    }
                    else  // LessOrEqual
                    {
                        A_sym(row_idx, col_idx) = pimpl_->A(i, j);
                        A_sym(row_idx, col_idx + 1) = -pimpl_->A(i, j);
                        row_idx++;
                    }
                }
                col_idx += 2;
                break;
            }
        }
    }

    // Заполняем вектор правых частей b_sym
    int row_idx = 0;
    for (int i = 0; i < m; ++i)
    {
        if (pimpl_->constraintTypes[i] == ConstraintType::Equal)
        {
            // A_i x = b_i превращаем в: A_i x <= b_i и -A_i x <= -b_i
            b_sym[row_idx] = pimpl_->b[i];
            b_sym[row_idx + 1] = -pimpl_->b[i];
            row_idx += 2;
        }
        else if (pimpl_->constraintTypes[i] == ConstraintType::GreaterOrEqual)
        {
            // A_i x >= b_i превращаем в -A_i x <= -b_i
            b_sym[row_idx] = -pimpl_->b[i];
            row_idx++;
        }
        else  // LessOrEqual
        {
            b_sym[row_idx] = pimpl_->b[i];
            row_idx++;
        }
    }

    // Определяем тип оптимизации
    // Симметричная форма: максимизация с <=
    bool maximize_sym = true;
    
    // Если исходная задача на минимизацию, инвертируем целевую функцию
    if (!pimpl_->maximize)
    {
        c_sym = -c_sym;
    }

    return std::make_unique<Symmetrical>(A_sym, b_sym, c_sym, maximize_sym);
}


std::unique_ptr<Canonical> Common::ToCanonical() const
{
    // Стратегия: делегируем через симметричную форму
    // Common → Symmetrical → Canonical
    // Это проще и избегает дублирования логики
    
    auto symmetrical = ToSymmetrical();
    return symmetrical->ToCanonical();
}



std::unique_ptr<Common> Common::GetDual() const
{
    // Правила построения двойственной задачи из общей формы
    // Прямая задача:
    // max/min c^T x
    // A_i x {<=, >=, =} b_i  (i-е ограничение)
    // x_j {свободная, >= 0, <= 0}  (j-я переменная)
    //
    // Двойственная задача:
    // min/max b^T y
    // A^T_j y {>=, <=, =} c_j  (j-е ограничение для j-й переменной прямой)
    // y_i {свободная, >= 0, <= 0}  (i-я переменная для i-го ограничения прямой)
    //
    // Правила соответствий:
    // max → min, min → max
    // Ограничение <= → переменная >= 0
    // Ограничение >= → переменная <= 0
    // Ограничение =  → свободная переменная
    // Переменная >= 0 → ограничение >=
    // Переменная <= 0 → ограничение <=
    // Свободная переменная → ограничение =

    int m = pimpl_->A.rows();  // Количество ограничений прямой = количество переменных двойственной
    int n = pimpl_->A.cols();  // Количество переменных прямой = количество ограничений двойственной

    // Транспонируем матрицу A
    Eigen::MatrixXd A_dual = pimpl_->A.transpose();
    
    // Меняем местами b и c
    Eigen::VectorXd b_dual = pimpl_->c;
    Eigen::VectorXd c_dual = pimpl_->b;
    
    // Типы переменных двойственной задачи (из ограничений прямой)
    std::vector<VariableType> variableTypes_dual(m);
    for (int i = 0; i < m; ++i)
    {
        switch (pimpl_->constraintTypes[i])
        {
            case ConstraintType::LessOrEqual:
                variableTypes_dual[i] = pimpl_->maximize ? VariableType::NonNegative : VariableType::NonPositive;
                break;
            case ConstraintType::GreaterOrEqual:
                variableTypes_dual[i] = pimpl_->maximize ? VariableType::NonPositive : VariableType::NonNegative;
                break;
            case ConstraintType::Equal:
                variableTypes_dual[i] = VariableType::Free;
                break;
        }
    }
    
    // Типы ограничений двойственной задачи (из переменных прямой)
    std::vector<ConstraintType> constraintTypes_dual(n);
    for (int j = 0; j < n; ++j)
    {
        switch (pimpl_->variableTypes[j])
        {
            case VariableType::NonNegative:
                constraintTypes_dual[j] = pimpl_->maximize ? ConstraintType::GreaterOrEqual : ConstraintType::LessOrEqual;
                break;
            case VariableType::NonPositive:
                constraintTypes_dual[j] = pimpl_->maximize ? ConstraintType::LessOrEqual : ConstraintType::GreaterOrEqual;
                break;
            case VariableType::Free:
                constraintTypes_dual[j] = ConstraintType::Equal;
                break;
        }
    }
    
    // Меняем тип оптимизации
    bool maximize_dual = !pimpl_->maximize;
    
    return std::make_unique<Common>(A_dual, b_dual, c_dual, constraintTypes_dual, variableTypes_dual, maximize_dual);
}
