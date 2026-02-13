#include <iostream>
#include <Eigen/Dense>

#include "ProblemTypes/Canonical.h"
#include "EnumerationSolver.h"

int main(){

//    Eigen::MatrixXd mat(4, 3);
//    mat << 2, 1, 1,
//           1, -1, 0,
//           3, -1, 2,
//           4, 2, 2;
//
//    Eigen::VectorXd b(4);
//    Eigen::VectorXd c(3);
//
//        b << 2, -2, 2, 4;
//
//    c << 1, 2, 3;
//
//    Canonical problem(mat, b, c);


// A : 4 x 10
//    Eigen::MatrixXd mat(4, 10);
//    mat <<  1,  2,  3,   1, -1,  2, -2,  1, 0, 0,
//            2, -1,  1,   4, -4, -1,  1,  0, 1, 0,
//            -1,  1,  2,   1, -1,  1, -1,  0, 0, 1,
//            1,  1,  1,  -1,  1,  3, -3,  0, 0, 0;
//
//    Eigen::VectorXd b(4);
//    b << 10, 8, 6, 7;
//
//    Eigen::VectorXd c(10);
//    c << 1, 2, 3, -1, 1, 4, -4, 0, 0, 0;


    Eigen::MatrixXd A(2,4);
    A << 4, 3, 0, 1,
            0, 4, 0, 4;

    Eigen::Vector2d b(4, 6);

    Eigen::Vector4d c(5, 1, 0, 0);
    Canonical problem(A, b, c);

    EnumerationSolver es(problem);
    es.solveEnumeration();
    std::cout<< es.findSolution(); // 1.25 !?!?!?!?!?!?!?!?!?!??!?!?!?!


    return 0;
}