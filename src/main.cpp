#include <iostream>
#include <Eigen/Dense>

#include "ProblemTypes/Canonical.h"
#include "EnumerationSolver.h"

class Vector2d;

int main(){

    Eigen::MatrixXd mat(4, 3);

//    mat << 1, 2,
//           2, 4;

    mat << 2, 1, 1,
           1, -1, 0,
           3, -1, 2,
           4, 2, 2; // число переменных в задаче зависит от числа ограничений (уравнений)

    Eigen::VectorXd b(4);
    Eigen::VectorXd c(3);

        b << 2, -2, 2, 4;

    c << 1, 2, 3;

    Canonical problem(mat, b, c);

    EnumerationSolver es(problem);
    es.solveEnumeration();
    return 0;
}