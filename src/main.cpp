#include <iostream>
#include <Eigen/Dense>
#include "SimplexSolover.h"

int main(){

    // Фиксированная матрица 2x2 (double)
    // Eigen::Matrix2d mat;
    // mat << 1, 2,
    //        3, 4;

    // // Фиксированный вектор из 2-х элементов
    // Eigen::Vector2d vec(5, 6);

    // // Матрично-векторное умножение
    // Eigen::Vector2d result = mat * vec;
    // std::cout << "Результат: " << result.transpose() << std::endl;

    Eigen::MatrixXd A(2,4);
    A << 4, 3, 0, 1,
        0, 4, 0, 4;

    Eigen::Vector2d b(4, 6);

    Eigen::Vector4d c(5, 1, 0, 0);

    Canonical problem(A, b, c);
    Solver s(problem);

    std::cout << s.solve().transpose() << std::endl;

    // std::cout << b;

    return 0;
}