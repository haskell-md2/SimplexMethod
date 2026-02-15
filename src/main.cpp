#include "CommonParser.h"
#include "ProblemTypes/Common.h"
#include "ProblemTypes/Symmetrical.h"
#include "ProblemTypes/Canonical.h"
#include "SimplexSolver.h"
#include "EnumerationSolver.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <unistd.h>  // для getcwd

namespace fs = std::filesystem;

void printUsage(const char* programName) {
    std::cout << "Использование:" << std::endl;
    std::cout << "  " << programName << " [опции]" << std::endl;
    std::cout << "\nОпции:" << std::endl;
    std::cout << "  -f, --file <путь>    Путь к файлу с задачей" << std::endl;
    std::cout << "  -h, --help           Показать справку" << std::endl;
    std::cout << "\nПримеры:" << std::endl;
    std::cout << "  " << programName << "                          # Использовать встроенный пример" << std::endl;
    std::cout << "  " << programName << " -f input.txt             # Файл из корня проекта" << std::endl;
    std::cout << "  " << programName << " -f data/problem.txt      # Относительный путь" << std::endl;
    std::cout << "  " << programName << " -f /absolute/path.txt    # Абсолютный путь" << std::endl;
}

std::string resolveFilePath(const std::string& filepath) {
    fs::path filePath(filepath);
    
    // Если путь абсолютный, используем как есть
    if (filePath.is_absolute()) {
        return filepath;
    }
    
    // Получаем текущую рабочую директорию (build/)
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        return filepath;
    }
    
    fs::path currentDir(cwd);
    
    // Проверяем, находимся ли мы в build/
    if (currentDir.filename() == "build") {
        // Поднимаемся на уровень выше
        fs::path projectRoot = currentDir.parent_path();
        fs::path fullPath = projectRoot / filePath;
        return fullPath.string();
    } else {
        // Мы уже в корне проекта
        fs::path fullPath = currentDir / filePath;
        return fullPath.string();
    }
}

int main(int argc, char* argv[])
{
    std::string inputFile;
    bool useFile = false;
    
    // Парсинг аргументов командной строки
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "-f" || arg == "--file") {
            if (i + 1 < argc) {
                inputFile = argv[i + 1];
                useFile = true;
                i++; // Пропускаем следующий аргумент
            } else {
                std::cerr << "Ошибка: опция " << arg << " требует аргумент" << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        }
        else {
            std::cerr << "Ошибка: неизвестная опция '" << arg << "'" << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       Simplex Method - Лабораторная работа 1              ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n" << std::endl;
    
    CommonParser parser;
    std::unique_ptr<Common> common;
    
    if (useFile)
    {
        // Преобразуем путь относительно корня проекта
        std::string resolvedPath = resolveFilePath(inputFile);
        
        std::cout << "Чтение из файла: " << resolvedPath << std::endl;
        
        // Проверяем существование файла
        if (!fs::exists(resolvedPath)) {
            std::cerr << "Ошибка: файл не найден: " << resolvedPath << std::endl;
            
            char cwd[1024];
            if (getcwd(cwd, sizeof(cwd)) != nullptr) {
                std::cerr << "Текущая директория: " << cwd << std::endl;
            }
            
            // Подсказка для пользователя
            std::cerr << "\nПопытка поиска файла в других местах..." << std::endl;
            
            // Попробуем найти в текущей директории
            if (fs::exists(inputFile)) {
                std::cerr << "Найден в текущей директории: " << inputFile << std::endl;
                resolvedPath = inputFile;
            } else {
                return 1;
            }
        }
        
        common = parser.ParseFromFile(resolvedPath);
    }
    else
    {
        // Используется встроенный пример
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
    Eigen::VectorXd initialSolution = canonical->GetBasicSolution();
    
    int numOriginalVars = common->GetObjectiveCoefficients().size();
    
    std::cout << "Исходные переменные (x₁, ..., x" << numOriginalVars << "):" << std::endl;
    int displayVars = std::min(numOriginalVars, static_cast<int>(initialSolution.size()));
    for (int i = 0; i < displayVars; ++i) {
        std::cout << "  x" << (i+1) << " = " 
                  << std::setw(8) << std::fixed << std::setprecision(2) 
                  << initialSolution[i] << std::endl;
    }
    
    double obj_value = canonical->Evaluate(initialSolution);
    bool feasible = canonical->IsFeasibleBasis();
    
    std::cout << "\nЗначение Z = " << std::fixed << std::setprecision(4) << obj_value << std::endl;
    std::cout << "Базис: " << (feasible ? "✓ ДОПУСТИМ" : "✗ НЕДОПУСТИМ") << std::endl;
    
    if (!feasible) {
        std::cout << "\n⚠  Необходим метод искусственного базиса!" << std::endl;
    }
    
    std::cout << "\n╚═══════════════════════════════════════════════════════════╝\n" << std::endl;
    
    
    // ============== РЕШЕНИЕ СИМПЛЕКС-МЕТОДОМ ==============
    std::cout << "\n┌─ 6. РЕШЕНИЕ СИМПЛЕКС-МЕТОДОМ ──────────────────────────┐\n" << std::endl;
    
    try {
        auto start_simplex = std::chrono::high_resolution_clock::now();
        
        Solver solver(*canonical);
        Eigen::VectorXd simplexSolution = solver.solve();
        
        auto end_simplex = std::chrono::high_resolution_clock::now();
        auto duration_simplex = std::chrono::duration_cast<std::chrono::milliseconds>(end_simplex - start_simplex);
        
        std::cout << "✓ Оптимальное решение найдено!\n" << std::endl;
        
        std::cout << "Оптимальные значения переменных:" << std::endl;
        for (int i = 0; i < numOriginalVars; ++i) {
            std::cout << "  x" << (i+1) << " = " 
                      << std::setw(10) << std::fixed << std::setprecision(4) 
                      << simplexSolution[i] << std::endl;
        }
        
        double simplexValue = 0.0;
        const Eigen::VectorXd& c_orig = common->GetObjectiveCoefficients();
        for (int i = 0; i < numOriginalVars; ++i) {
            simplexValue += c_orig[i] * simplexSolution[i];
        }
        
        std::cout << "\nОптимальное значение целевой функции:" << std::endl;
        std::cout << "  Z* = " << std::fixed << std::setprecision(6) << simplexValue << std::endl;
        std::cout << "\nВремя выполнения: " << duration_simplex.count() << " мс" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Ошибка при решении: " << e.what() << std::endl;
    }
    
    std::cout << "\n╚═══════════════════════════════════════════════════════════╝\n" << std::endl;
    
    
    // ============== РЕШЕНИЕ МЕТОДОМ ПЕРЕБОРА ВЕРШИН ==============
    std::cout << "\n┌─ 7. РЕШЕНИЕ МЕТОДОМ ПЕРЕБОРА ВЕРШИН ───────────────────┐\n" << std::endl;

    try {
        auto start_enum = std::chrono::high_resolution_clock::now();
        
        EnumerationSolver enumSolver(*canonical);
        enumSolver.solveEnumeration();
        
        std::cout << "\n" << std::endl;
        double enumValue = enumSolver.findSolution();
        
        auto end_enum = std::chrono::high_resolution_clock::now();
        auto duration_enum = std::chrono::duration_cast<std::chrono::milliseconds>(end_enum - start_enum);
        
        std::cout << "\n✓ Оптимальное решение найдено!\n" << std::endl;
        
        // Получаем лучшую вершину в формате симметричной задачи
        Eigen::VectorXd enumSolutionSymmetric = enumSolver.getBestVertex();
        
        // Преобразуем из симметричной формы обратно в исходную
        // Нужно знать, какие переменные были расщеплены
        const auto& varTypes = common->GetVariableTypes();
        
        Eigen::VectorXd enumSolution(numOriginalVars);
        int symIdx = 0;
        
        for (int i = 0; i < numOriginalVars; ++i) {
            if (varTypes[i] == Common::VariableType::Free) {
                // Свободная переменная была расщеплена: xi = xi' - xi''
                if (symIdx + 1 < enumSolutionSymmetric.size()) {
                    enumSolution[i] = enumSolutionSymmetric[symIdx] - enumSolutionSymmetric[symIdx + 1];
                    symIdx += 2;
                } else {
                    enumSolution[i] = 0.0;
                }
            } else if (varTypes[i] == Common::VariableType::NonPositive) {
                // Неположительная переменная: xi = -xi'
                if (symIdx < enumSolutionSymmetric.size()) {
                    enumSolution[i] = -enumSolutionSymmetric[symIdx];
                    symIdx += 1;
                } else {
                    enumSolution[i] = 0.0;
                }
            } else {
                // Неотрицательная переменная: xi = xi
                if (symIdx < enumSolutionSymmetric.size()) {
                    enumSolution[i] = enumSolutionSymmetric[symIdx];
                    symIdx += 1;
                } else {
                    enumSolution[i] = 0.0;
                }
            }
        }
        
        std::cout << "Оптимальные значения переменных:" << std::endl;
        for (int i = 0; i < numOriginalVars; ++i) {
            std::cout << "  x" << (i+1) << " = " 
                    << std::setw(10) << std::fixed << std::setprecision(4) 
                    << enumSolution[i] << std::endl;
        }
        
        std::cout << "\nОптимальное значение целевой функции:" << std::endl;
        std::cout << "  Z* = " << std::fixed << std::setprecision(6) << enumValue << std::endl;
        std::cout << "\nВремя выполнения: " << duration_enum.count() << " мс" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Ошибка при решении: " << e.what() << std::endl;
    }

    std::cout << "\n╚═══════════════════════════════════════════════════════════╝\n" << std::endl;


    return 0;
}
