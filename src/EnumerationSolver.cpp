
#include "EnumerationSolver.h"

#include <Eigen/Dense>
#include <Eigen/QR>
#include <iostream>
#include <vector>


using namespace Eigen;

EnumerationSolver::EnumerationSolver(const Canonical &problem) : problem(problem){};

void combine_indices(int n, int k, int start, std::vector<int>& current, std::vector<std::vector<int>>& result) {
    if (current.size() == k) {
        result.push_back(current);
        return;
    }
    for (int i = start; i < n; ++i) {
        current.push_back(i);
        combine_indices(n, k, i + 1, current, result);
        current.pop_back();
    }
}

std::vector<std::vector<int>> get_index_combinations(int n, int k) {
    std::vector<std::vector<int>> result;
    if (k < 0 || k > n) return result;
    std::vector<int> current;
    combine_indices(n, k, 0, current, result);
    return result;
}

bool EnumerationSolver::handleSolution(const Eigen::VectorXd &xB, int rank, int A_num_cols,
                                                 const std::vector<int> & cols){

    const MatrixXd &A = problem._A;
    const VectorXd &b = problem._b;

    //  проверка на NaN/Inf
    bool okFinite = true;
    for (int i = 0; i < xB.size(); ++i) {
        if (!std::isfinite(xB(i))) { okFinite = false; break; }
    }
    if (!okFinite) return true;

    // 4) восстановление полного вектора
    Eigen::VectorXd x_full = Eigen::VectorXd::Zero(A_num_cols);
    for (int j = 0; j < rank; ++j) x_full(cols[j]) = xB(j); // обнуляем столбцы до квадратной матрицы

    // 5) проверка допустимости (неотрицательность)
    if (x_full.minCoeff() < -tol_feas) return true;

    // 6) проверка выполнения всех уравнений A*x = b (по всей системе)
    double rel_res = (A * x_full - b).norm();
    if (rel_res > tol_res) return true; // САМАЯ СОМНИТЕЛЬНАЯ ПРОВЕРКА, ОНА НУЖНА, Т К ТУТ МУДАЦКИЙ СПОСОБ ПРОВЕРКИ НЕЗАВИСИМОСТИ СТРОК

    // 7) дедупликация (если уже есть похожая вершина, пропускаем)
    bool duplicate = false;
    for (const auto &v : found_vertices) {
        if ((v - x_full).norm() < tol_dup) { duplicate = true; break; }
    }
    if (duplicate) return true;

    found_vertices.push_back(x_full);

    return false;
}


double EnumerationSolver::findSolution(){

    double best = std::numeric_limits<double>::max();

    for(auto &x_full: found_vertices){
        double val = problem._c.dot(x_full);
        if(val < best) best = val;
        std::cout << "Feasible vertex: x = " << x_full.transpose() << "  f = " << val << std::endl;
    }
    return best;
}

void EnumerationSolver::solveEnumeration() {
    const MatrixXd &A = problem._A;
    const VectorXd &b = problem._b;

    ColPivHouseholderQR<MatrixXd> qr(A.transpose());  // транспонируем, так как qr.colsPermutation() умеет находить только независимые столбцы
    int rank = qr.rank();
    int A_num_cols = (int)A.cols();

    VectorXi perm = qr.colsPermutation().indices();
    std::vector<int> independent_rows;
    for (int i = 0; i < rank; ++i) independent_rows.push_back(perm(i));

    MatrixXd B((Eigen::Index)rank, A_num_cols);
    VectorXd b_sub((Eigen::Index)rank);
    for (int i = 0; i < rank; ++i) {
        B.row(i) = A.row(independent_rows[i]); // новая матрица с независимыми строками
        b_sub(i) = b(independent_rows[i]);  // соотв B вектор
    }

    std::vector<std::vector<int>> col_combs = get_index_combinations(A_num_cols, rank);

    for (size_t comb_idx = 0; comb_idx < col_combs.size(); ++comb_idx) {
        const auto &cols = col_combs[comb_idx];

        MatrixXd B_cols((Eigen::Index)rank, (Eigen::Index)rank);
        for (int j = 0; j < rank; ++j) B_cols.col(j) = B.col(cols[j]);

        // 1) проверим ранг базисной матрицы
        Eigen::FullPivLU<MatrixXd> fplu(B_cols);
        if (fplu.rank() < rank) continue; // вырожденный базис

        // 2) решаем быстро (PartialPivLU)
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(B_cols);
        Eigen::VectorXd xB = lu.solve(b_sub);

        handleSolution(xB, rank, A_num_cols, cols); // добавляет вершину в found_vertices
    }
}



EnumerationSolver::~EnumerationSolver() = default;

