#include "SymmetricalParser.h"
#include "ProblemTypes/Common.h"
#include "ProblemTypes/Symmetrical.h"
#include "ProblemTypes/Canonical.h"
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[])
{
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       Simplex Method - Лабораторная работа 1              ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n" << std::endl;
    
    // Задача из условия
    Eigen::MatrixXd A(4, 5);
    A << 2, -2,  3, -1,  2,
         5, -3,  2,  3, -1,
        -2,  3,  3, -3,  1,
        -1, -4,  3,  1, -4;
    
    Eigen::VectorXd b(4);
    b << 5, -46, 53, -57;
    
    Eigen::VectorXd c(5);
    c << -1, -5, 5, 1, 4;
    
    std::vector<Common::ConstraintType> constraintTypes = {
        Common::ConstraintType::LessOrEqual,
        Common::ConstraintType::LessOrEqual,
        Common::ConstraintType::GreaterOrEqual,
        Common::ConstraintType::Equal
    };
    
    std::vector<Common::VariableType> variableTypes = {
        Common::VariableType::NonNegative,
        Common::VariableType::NonNegative,
        Common::VariableType::NonNegative,
        Common::VariableType::Free,
        Common::VariableType::Free
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
    
    return 0;
}
