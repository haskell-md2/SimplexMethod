#include "SymmetricalParser.h"
#include <iostream>

int main()
{
    SymmetricalParser parser;
    
    auto symmetrical = parser.ParseFromFile("input_symmetric.txt");
    
    if (!symmetrical)
    {
        std::cerr << "Ошибка: " << parser.GetLastError() << std::endl;
        return 1;
    }
    
    // Выводим симметричную форму
    symmetrical->Print();
    
    // Получаем двойственную задачу
    auto dual = symmetrical->GetDual();
    std::cout << "\n\n";
    dual->Print();
    
    // Преобразуем в каноническую
    auto canonical = symmetrical->ToCanonical();
    std::cout << "\n\n";
    canonical->Print();
    
    return 0;
}
