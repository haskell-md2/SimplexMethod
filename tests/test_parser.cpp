#include <gtest/gtest.h>
#include "SymmetricalParser.h"

TEST(ParserTest, ParseFromStringMaximization)
{
    std::string input = R"(
        maximize
        
        objective:
        3 5
        
        constraints:
        1 2 10
        3 4 20
    )";
    
    SymmetricalParser parser;
    auto sym = parser.ParseFromString(input);
    
    ASSERT_NE(sym, nullptr);
    EXPECT_TRUE(sym->IsMaximization());
    EXPECT_EQ(sym->GetConstraintsMatrix().rows(), 2);
    EXPECT_EQ(sym->GetConstraintsMatrix().cols(), 2);
}

TEST(ParserTest, ParseFromStringMinimization)
{
    std::string input = R"(
        minimize
        
        objective:
        7 8
        
        constraints:
        1 1 5
        2 3 12
    )";
    
    SymmetricalParser parser;
    auto sym = parser.ParseFromString(input);
    
    ASSERT_NE(sym, nullptr);
    EXPECT_FALSE(sym->IsMaximization());
}

TEST(ParserTest, ParseWithComments)
{
    std::string input = R"(
        # Это комментарий
        maximize
        
        # Целевая функция
        objective:
        1 2 3  # еще комментарий
        
        constraints:
        1 0 0 5  # первое ограничение
        0 1 0 6  # второе
        0 0 1 7  # третье
    )";
    
    SymmetricalParser parser;
    auto sym = parser.ParseFromString(input);
    
    ASSERT_NE(sym, nullptr);
    EXPECT_EQ(sym->GetObjectiveCoefficients().size(), 3);
}

TEST(ParserTest, InvalidInput)
{
    std::string input = R"(
        maximize
        # Нет objective и constraints
    )";
    
    SymmetricalParser parser;
    auto sym = parser.ParseFromString(input);
    
    EXPECT_EQ(sym, nullptr);
    EXPECT_FALSE(parser.GetLastError().empty());
}
