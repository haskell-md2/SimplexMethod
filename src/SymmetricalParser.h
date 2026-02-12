#pragma once

#include "ProblemTypes/Symmetrical.h"
#include <string>
#include <memory>
#include <fstream>
#include <sstream>

/**
 * @brief Парсер для чтения задач линейного программирования в симметричной форме
 * Создает объект Symmetrical (симметричная форма задачи)
 */
class SymmetricalParser
{
public:
    SymmetricalParser() = default;
    ~SymmetricalParser() = default;

    /**
     * @brief Парсить задачу из файла
     * @param filename Имя файла с задачей
     * @return Умный указатель на объект Symmetrical или nullptr при ошибке
     */
    std::unique_ptr<Symmetrical> ParseFromFile(const std::string& filename);

    /**
     * @brief Парсить задачу из строки
     * @param content Строка с описанием задачи
     * @return Умный указатель на объект Symmetrical или nullptr при ошибке
     */
    std::unique_ptr<Symmetrical> ParseFromString(const std::string& content);

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
     * @return Умный указатель на объект Symmetrical или nullptr при ошибке
     */
    std::unique_ptr<Symmetrical> ParseFromStream(std::istream& stream);

    /**
     * @brief Удалить пробелы и комментарии из строки
     * @param line Исходная строка
     * @return Очищенная строка
     */
    std::string TrimAndRemoveComments(const std::string& line);
};
