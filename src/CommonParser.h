#pragma once

#include "ProblemTypes/Common.h"
#include <string>
#include <memory>
#include <fstream>
#include <sstream>

/**
 * @brief Парсер для чтения задач линейного программирования в общей форме
 * Создает объект Common (общая форма задачи с произвольными ограничениями)
 */
class CommonParser
{
public:
    CommonParser() = default;
    ~CommonParser() = default;

    /**
     * @brief Парсить задачу из файла
     * @param filename Имя файла с задачей
     * @return Умный указатель на объект Common или nullptr при ошибке
     */
    std::unique_ptr<Common> ParseFromFile(const std::string& filename);

    /**
     * @brief Парсить задачу из строки
     * @param content Строка с описанием задачи
     * @return Умный указатель на объект Common или nullptr при ошибке
     */
    std::unique_ptr<Common> ParseFromString(const std::string& content);

    /**
     * @brief Получить последнюю ошибку парсинга
     * @return Строка с описанием ошибки
     */
    std::string GetLastError() const;

private:
    std::string lastError_;

    /**
     * @brief Внутренний метод парсинга из потока
     * @param stream Входной поток
     * @return Умный указатель на объект Common или nullptr при ошибке
     */
    std::unique_ptr<Common> ParseFromStream(std::istream& stream);

    /**
     * @brief Удалить пробелы и комментарии из строки
     * @param line Исходная строка
     * @return Очищенная строка
     */
    std::string TrimAndRemoveComments(const std::string& line);

    /**
     * @brief Парсить тип ограничения из строки
     * @param token Токен (<=, >=, =)
     * @return Тип ограничения
     */
    Common::ConstraintType ParseConstraintType(const std::string& token);

    /**
     * @brief Парсить тип переменной из строки
     * @param token Токен (>=, <=, free, R)
     * @return Тип переменной
     */
    Common::VariableType ParseVariableType(const std::string& token);
};
