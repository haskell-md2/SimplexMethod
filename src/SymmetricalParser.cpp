#include "SymmetricalParser.h"
#include <algorithm>
#include <vector>

std::unique_ptr<Symmetrical> SymmetricalParser::ParseFromFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        lastError_ = "Не удалось открыть файл: " + filename;
        return nullptr;
    }

    return ParseFromStream(file);
}

std::unique_ptr<Symmetrical> SymmetricalParser::ParseFromString(const std::string& content)
{
    std::istringstream stream(content);
    return ParseFromStream(stream);
}

std::string SymmetricalParser::GetLastError() const
{
    return lastError_;
}

std::string SymmetricalParser::TrimAndRemoveComments(const std::string& line)
{
    // Удаляем комментарии (все после #)
    size_t commentPos = line.find('#');
    std::string result = (commentPos != std::string::npos) ? line.substr(0, commentPos) : line;

    // Удаляем пробелы в начале и конце
    auto start = result.find_first_not_of(" \t\r\n");
    auto end = result.find_last_not_of(" \t\r\n");

    if (start == std::string::npos)
        return "";

    return result.substr(start, end - start + 1);
}

std::unique_ptr<Symmetrical> SymmetricalParser::ParseFromStream(std::istream& stream)
{
    try
    {
        std::string line;
        bool maximize = true;
        std::vector<double> objectiveCoeffs;
        std::vector<std::vector<double>> constraintCoeffs;
        std::vector<double> rightHandSides;

        enum class Section { None, Objective, Constraints };
        Section currentSection = Section::None;

        while (std::getline(stream, line))
        {
            line = TrimAndRemoveComments(line);
            if (line.empty())
                continue;

            // Определяем секцию
            if (line == "maximize" || line == "max")
            {
                maximize = true;
                continue;
            }
            else if (line == "minimize" || line == "min")
            {
                maximize = false;
                continue;
            }
            else if (line == "objective:" || line == "objective")
            {
                currentSection = Section::Objective;
                continue;
            }
            else if (line == "constraints:" || line == "constraints" || line == "subject to:" || line == "subject to")
            {
                currentSection = Section::Constraints;
                continue;
            }

            // Парсим содержимое секции
            std::istringstream lineStream(line);

            switch (currentSection)
            {
                case Section::Objective:
                {
                    double coeff;
                    while (lineStream >> coeff)
                    {
                        objectiveCoeffs.push_back(coeff);
                    }
                    break;
                }

                case Section::Constraints:
                {
                    // Формат: коэффициенты правая_часть
                    // Для максимизации подразумеваем <=
                    // Для минимизации подразумеваем >=
                    // Например: 2 3 1 18  означает  2*x1 + 3*x2 + 1*x3 <= 18 (для max)
                    
                    std::vector<double> coeffs;
                    double value;
                    
                    while (lineStream >> value)
                    {
                        coeffs.push_back(value);
                    }

                    if (coeffs.size() < 2)
                    {
                        lastError_ = "Недостаточно данных в ограничении: " + line;
                        return nullptr;
                    }

                    // Последний элемент - правая часть
                    double rhs = coeffs.back();
                    coeffs.pop_back();

                    constraintCoeffs.push_back(coeffs);
                    rightHandSides.push_back(rhs);
                    break;
                }

                case Section::None:
                    lastError_ = "Данные вне секции: " + line;
                    return nullptr;
            }
        }

        // Валидация
        if (objectiveCoeffs.empty())
        {
            lastError_ = "Целевая функция не задана";
            return nullptr;
        }

        if (constraintCoeffs.empty())
        {
            lastError_ = "Ограничения не заданы";
            return nullptr;
        }

        int numVars = objectiveCoeffs.size();

        // Проверяем размерность ограничений
        for (const auto& constraint : constraintCoeffs)
        {
            if (static_cast<int>(constraint.size()) != numVars)
            {
                lastError_ = "Несоответствие размерности ограничений и целевой функции";
                return nullptr;
            }
        }

        // Создаем матрицы Eigen
        int m = constraintCoeffs.size();
        int n = numVars;

        Eigen::MatrixXd A(m, n);
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                A(i, j) = constraintCoeffs[i][j];
            }
        }

        Eigen::VectorXd b(m);
        for (int i = 0; i < m; ++i)
        {
            b[i] = rightHandSides[i];
        }

        Eigen::VectorXd c(n);
        for (int i = 0; i < n; ++i)
        {
            c[i] = objectiveCoeffs[i];
        }

        return std::make_unique<Symmetrical>(A, b, c, maximize);
    }
    catch (const std::exception& e)
    {
        lastError_ = std::string("Ошибка парсинга: ") + e.what();
        return nullptr;
    }
}
