#include "CommonParser.h"
#include <algorithm>
#include <vector>
#include <cctype>

std::unique_ptr<Common> CommonParser::ParseFromFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        lastError_ = "Не удалось открыть файл: " + filename;
        return nullptr;
    }

    return ParseFromStream(file);
}

std::unique_ptr<Common> CommonParser::ParseFromString(const std::string& content)
{
    std::istringstream stream(content);
    return ParseFromStream(stream);
}

std::string CommonParser::GetLastError() const
{
    return lastError_;
}

std::string CommonParser::TrimAndRemoveComments(const std::string& line)
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

Common::ConstraintType CommonParser::ParseConstraintType(const std::string& token)
{
    if (token == "<=" || token == "≤")
        return Common::ConstraintType::LessOrEqual;
    else if (token == ">=" || token == "≥")
        return Common::ConstraintType::GreaterOrEqual;
    else if (token == "=" || token == "==")
        return Common::ConstraintType::Equal;
    else
        throw std::runtime_error("Неизвестный тип ограничения: " + token);
}

Common::VariableType CommonParser::ParseVariableType(const std::string& token)
{
    std::string lower = token;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == ">=" || lower == ">=0" || lower == ">= 0" || lower == "nonnegative")
        return Common::VariableType::NonNegative;
    else if (lower == "<=" || lower == "<=0" || lower == "<= 0" || lower == "nonpositive")
        return Common::VariableType::NonPositive;
    else if (lower == "free" || lower == "r" || lower == "∈r" || lower == "unrestricted")
        return Common::VariableType::Free;
    else
        throw std::runtime_error("Неизвестный тип переменной: " + token);
}

std::unique_ptr<Common> CommonParser::ParseFromStream(std::istream& stream)
{
    try
    {
        std::string line;
        bool maximize = true;
        std::vector<double> objectiveCoeffs;
        std::vector<std::vector<double>> constraintCoeffs;
        std::vector<double> rightHandSides;
        std::vector<Common::ConstraintType> constraintTypes;
        std::vector<Common::VariableType> variableTypes;

        enum class Section { None, Objective, Constraints, Variables };
        Section currentSection = Section::None;

        while (std::getline(stream, line))
        {
            line = TrimAndRemoveComments(line);
            if (line.empty())
                continue;

            // Определяем секцию
            std::string lowerLine = line;
            std::transform(lowerLine.begin(), lowerLine.end(), lowerLine.begin(), ::tolower);

            if (lowerLine == "maximize" || lowerLine == "max")
            {
                maximize = true;
                continue;
            }
            else if (lowerLine == "minimize" || lowerLine == "min")
            {
                maximize = false;
                continue;
            }
            else if (lowerLine == "objective:" || lowerLine == "objective")
            {
                currentSection = Section::Objective;
                continue;
            }
            else if (lowerLine == "constraints:" || lowerLine == "constraints" || 
                     lowerLine == "subject to:" || lowerLine == "subject to")
            {
                currentSection = Section::Constraints;
                continue;
            }
            else if (lowerLine == "variables:" || lowerLine == "variables" ||
                     lowerLine == "bounds:" || lowerLine == "bounds")
            {
                currentSection = Section::Variables;
                continue;
            }

            // Парсим содержимое секции
            switch (currentSection)
            {
                case Section::Objective:
                {
                    std::istringstream lineStream(line);
                    double coeff;
                    while (lineStream >> coeff)
                    {
                        objectiveCoeffs.push_back(coeff);
                    }
                    break;
                }

                case Section::Constraints:
                {
                    // Формат: коэффициенты тип_ограничения правая_часть
                    // Например: 2 3 1 <= 18
                    //          -1 2 >= 5
                    //           3 4 = 12
                    
                    std::istringstream lineStream(line);
                    std::vector<double> coeffs;
                    std::vector<std::string> tokens;
                    std::string token;
                    
                    // Читаем все токены
                    while (lineStream >> token)
                    {
                        tokens.push_back(token);
                    }

                    if (tokens.size() < 3)
                    {
                        lastError_ = "Недостаточно данных в ограничении: " + line;
                        return nullptr;
                    }

                    // Ищем оператор сравнения (<=, >=, =)
                    size_t opIndex = tokens.size();
                    for (size_t i = 0; i < tokens.size(); ++i)
                    {
                        if (tokens[i] == "<=" || tokens[i] == ">=" || 
                            tokens[i] == "=" || tokens[i] == "==" ||
                            tokens[i] == "≤" || tokens[i] == "≥")
                        {
                            opIndex = i;
                            break;
                        }
                    }

                    if (opIndex == tokens.size() || opIndex == 0 || opIndex == tokens.size() - 1)
                    {
                        lastError_ = "Некорректный формат ограничения: " + line;
                        return nullptr;
                    }

                    // Коэффициенты (до оператора)
                    for (size_t i = 0; i < opIndex; ++i)
                    {
                        try {
                            coeffs.push_back(std::stod(tokens[i]));
                        } catch (const std::exception& e) {
                            lastError_ = "Ошибка преобразования числа '" + tokens[i] + "': " + e.what();
                            return nullptr;
                        }
                    }

                    // Тип ограничения
                    Common::ConstraintType cType = ParseConstraintType(tokens[opIndex]);

                    // Правая часть (после оператора)
                    double rhs;
                    try {
                        rhs = std::stod(tokens[opIndex + 1]);
                    } catch (const std::exception& e) {
                        lastError_ = "Ошибка преобразования правой части '" + tokens[opIndex + 1] + "': " + e.what();
                        return nullptr;
                    }

                    constraintCoeffs.push_back(coeffs);
                    rightHandSides.push_back(rhs);
                    constraintTypes.push_back(cType);
                    break;
                }

                case Section::Variables:
                {
                    // Формат: x1 >= 0
                    //         x2 free
                    //         x3 <= 0
                    std::istringstream lineStream(line);
                    std::string varName, typeToken;
                    
                    lineStream >> varName; // Читаем имя переменной (x1, x2...)
                    
                    // Читаем остальное как тип
                    std::string rest;
                    std::getline(lineStream, rest);
                    rest = TrimAndRemoveComments(rest);
                    
                    if (rest.empty())
                    {
                        // По умолчанию >= 0
                        variableTypes.push_back(Common::VariableType::NonNegative);
                    }
                    else
                    {
                        variableTypes.push_back(ParseVariableType(rest));
                    }
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

        int numVars = static_cast<int>(objectiveCoeffs.size());

        // Проверяем размерность ограничений
        for (size_t i = 0; i < constraintCoeffs.size(); ++i)
        {
            if (static_cast<int>(constraintCoeffs[i].size()) != numVars)
            {
                lastError_ = "Несоответствие размерности ограничения " + std::to_string(i+1) + 
                            " (ожидается " + std::to_string(numVars) + " переменных, получено " + 
                            std::to_string(constraintCoeffs[i].size()) + ")";
                return nullptr;
            }
        }

        // Если типы переменных не заданы, по умолчанию >= 0
        if (variableTypes.empty())
        {
            variableTypes.resize(numVars, Common::VariableType::NonNegative);
        }
        else if (static_cast<int>(variableTypes.size()) != numVars)
        {
            lastError_ = "Число типов переменных (" + std::to_string(variableTypes.size()) + 
                        ") не соответствует числу переменных (" + std::to_string(numVars) + ")";
            return nullptr;
        }

        // Создаем матрицы Eigen
        int m = static_cast<int>(constraintCoeffs.size());
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

        return std::make_unique<Common>(A, b, c, constraintTypes, variableTypes, maximize);
    }
    catch (const std::exception& e)
    {
        lastError_ = std::string("Ошибка парсинга: ") + e.what();
        return nullptr;
    }
}
