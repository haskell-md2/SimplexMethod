#include "CommonParser.h"
#include "ProblemTypes/Common.h"
#include "ProblemTypes/Symmetrical.h"
#include "ProblemTypes/Canonical.h"
#include "SimplexSolver.h"
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[])
{
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       Simplex Method - Лабораторная работа 1              ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n" << std::endl;
    
    CommonParser parser;
    std::unique_ptr<Common> common;
    
    if (argc > 1)
    {
        // Читаем из файла
        std::string filename = argv[1];
        std::cout << "Чтение из файла: " << filename << std::endl;
        common = parser.ParseFromFile(filename);
    }
    else
    {
        // Используем встроенный пример
        std::cout << "Используется встроенный пример" << std::endl;
        std::string input = R"(
            maximize
            
            objective:
            -1 -5 5 1 4
            
            constraints:
            2 -2 3 -1 2 <= 5
            5 -3 2 3 -1 <= -46
            -2 3 3 -3 1 >= 53
            -1 -4 3 1 -4 = -57
            
            variables:
            x1 >= 0
            x2 >= 0
            x3 >= 0
            x4 free
            x5 free
        )";
        
        common = parser.ParseFromString(input);
    }
    
    if (!common)
    {
        std::cerr << "Ошибка парсинга: " << parser.GetLastError() << std::endl;
        return 1;
    }
    
    std::cout << "┌─ 1. ИСХОДНАЯ ЗАДАЧА ───────────────────────────────────┐\n" << std::endl;
    common->Print();
    
    std::cout << "\n\n┌─ 2. СИММЕТРИЧНАЯ ФОРМА ────────────────────────────────┐\n" << std::endl;
    auto symmetrical = common->ToSymmetrical();
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
    int origVars = std::min(5, static_cast<int>(solution.size()));
    for (int i = 0; i < origVars; ++i) {
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
    
    // Решение симплекс-методом
    Solver s(*canonical);
    std::cout << "Оптимальное решение: " << s.solve().transpose() << std::endl;

    return 0;
}
