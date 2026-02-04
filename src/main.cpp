#include <iostream>
#include <Eigen/Dense>

int main(){

    // Фиксированная матрица 2x2 (double)
    Eigen::Matrix2d mat;
    mat << 1, 2,
           3, 4;

    // Фиксированный вектор из 2-х элементов
    Eigen::Vector2d vec(5, 6);

    // Матрично-векторное умножение
    Eigen::Vector2d result = mat * vec;
    std::cout << "Результат: " << result.transpose() << std::endl;

    return 0;
}