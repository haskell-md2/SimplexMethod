#include "SymmetricalParser.h"
#include "ProblemTypes/Common.h"
#include "ProblemTypes/Symmetrical.h"
#include "ProblemTypes/Canonical.h"
#include <iostream>
#include <Eigen/Dense>
#include "SimplexSolover.h"

// int main(){

//     // Фиксированная матрица 2x2 (double)
//     // Eigen::Matrix2d mat;
//     // mat << 1, 2,
//     //        3, 4;

//     // // Фиксированный вектор из 2-х элементов
//     // Eigen::Vector2d vec(5, 6);

//     // // Матрично-векторное умножение
//     // Eigen::Vector2d result = mat * vec;
//     // std::cout << "Результат: " << result.transpose() << std::endl;


//     Eigen::MatrixXd A(2,4);
//     A << 4, 3, 0, 1,
//         0, 4, 0, 4;

//     Eigen::Vector2d b(4, 6);

//     Eigen::Vector4d c(5, 1, 0, 0);

//     Canonical problem(A, b, c,{0,2});
//     Solver s(problem);

//     std::cout << s.solve().transpose() << std::endl;

//     std::cout << b;
// }
#include <iomanip>

int main(int argc, char* argv[])
{
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       Simplex Method - Лабораторная работа 1              ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n" << std::endl;
    
    // Задача из условия
    Eigen::MatrixXd A(2, 3);
    A << 1, 1, 1,
         2, 1, 0;
    
    Eigen::VectorXd b(2);
    b << 6,   // x1 + x2 + x3 ≤ 6
         8;   // 2x1 + x2 ≤ 8
    
    Eigen::VectorXd c(3);
    c << 3, 2, 4;   // MAX 3x1 + 2x2 + 4x3
    
    std::vector<Common::ConstraintType> constraintTypes = {
        Common::ConstraintType::LessOrEqual,
        Common::ConstraintType::LessOrEqual
    };
    
    std::vector<Common::VariableType> variableTypes = {
        Common::VariableType::NonNegative,
        Common::VariableType::NonNegative,
        Common::VariableType::NonNegative
    };
    
    Common common(A, b, c, constraintTypes, variableTypes, true);
    
    std::cout << "┌─ 1. ИСХОДНАЯ ЗАДАЧА ───────────────────────────────────┐\n" << std::endl;
    common.Print();
    
    std::cout << "\n\n┌─ 2. СИММЕТРИЧНАЯ ФОРМА ────────────────────────────────┐\n" << std::endl;
    auto symmetrical = common.ToSymmetrical();
    symmetrical->Print();
    
    std::cout << "\n\n┌─ 3. ДВОЙСТВЕННАЯ ЗАДАЧА ───────────────────────────────┐\n" << std::endl;
    auto dual = symmetrical->GetDual();
    dual->Print();
    
    std::cout << "\n\n┌─ 4. КАНОНИЧЕСКАЯ ФОРМА ────────────────────────────────┐\n" << std::endl;
    auto canonical = symmetrical->ToCanonical();
    canonical->Print();
    
    std::cout << "\n\n┌─ 5. НАЧАЛЬНОЕ РЕШЕНИЕ ─────────────────────────────────┐\n" << std::endl;
    Eigen::VectorXd solution = canonical->GetBasicSolution();
    
    std::cout << "Исходные переменные (x₁, ..., x₅):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  x" << (i+1) << " = " 
                  << std::setw(8) << std::fixed << std::setprecision(2) 
                  << solution[i] << std::endl;
    }
    
    double obj_value = canonical->Evaluate(solution);
    bool feasible = canonical->IsFeasibleBasis();
    
    std::cout << "\nЗначение Z = " << std::fixed << std::setprecision(4) << obj_value << std::endl;
    std::cout << "Базис: " << (feasible ? "✓ ДОПУСТИМ" : "✗ НЕДОПУСТИМ") << std::endl;
    
    if (!feasible) {
        std::cout << "\n⚠  Необходим метод искусственного базиса!" << std::endl;
    }
    
    std::cout << "\n╚═══════════════════════════════════════════════════════════╝\n" << std::endl;
    


    Solver s(*canonical);

    std::cout << s.solve().transpose() << std::endl;



    return 0;
}
