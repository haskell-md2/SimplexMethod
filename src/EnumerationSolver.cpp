//
// Created by ilya on 2/10/26.
//
#include "EnumerationSolver.h"

#include <Eigen/Dense>
#include <Eigen/QR>    // для ColPivHouseholderQR
#include <Eigen/SVD>   // опционально для SVD
#include <iostream>
#include <vector>
#include <stdexcept>


using namespace Eigen;

EnumerationSolver::EnumerationSolver(const Canonical &problem) : problem(problem){};



// Возвращает матрицу B (r x n), где r = rank(A) и строки B — независимые строки A.
// Порядок строк соответствует порядку, который дал pivoted-QR (можно отсортировать, если нужно исходный порядок).
MatrixXd extractIndependentRows(const Eigen::MatrixXd &A) {
    using namespace Eigen;
    if (A.rows() == 0) return A; // пустая матрица

    // Делаем pivoted QR от A.transpose(): независимоcть строк A <-> независимость столбцов A^T
    ColPivHouseholderQR<MatrixXd> qr(A.transpose());
    int r = qr.rank();

    VectorXi perm = qr.colsPermutation().indices(); // индексы колонок (т.е. индексы строк A)
    Eigen::MatrixXd B(r, A.cols());

    for (int i = 0; i < r; ++i) {
        B.row(i) = A.row( perm(i) );
    }
    return B;
}


#include <vector>

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

std::vector<int> reverse_index_list(const std::vector<int>& indexes, int n){
    std::vector<int> res;
    if (n <= 0) return res;

    std::vector<int> sorted = indexes;
    std::sort(sorted.begin(), sorted.end());

    int pos = 0;
    for (int i = 0; i < n; i++) {

        if (pos < (int)sorted.size() && i == sorted[pos]) {
            ++pos;
            continue;
        }
        res.push_back(i);
    }
    return res;
}


Eigen::VectorXd solve_linear_system(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    if (A.rows() != b.size()) {
        throw std::invalid_argument("A.rows() must equal b.size()");
    }

    // SVD с вычислением унитарных матриц U и V (необходимо для solve)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Встроенный метод solve() уже использует порог отсечения сингулярных чисел
    // (по умолчанию: max(rows,cols) * epsilon * max_singular_value)
    Eigen::VectorXd x = svd.solve(b);

    // Дополнительная проверка: оцениваем невязку для обнаружения сильной несовместности
    double residual = (A * x - b).norm();
    double b_norm = b.norm();
    if (b_norm > 0 && residual / b_norm > 1e-9) { // порог можно подстроить
        // Можно просто вернуть x с предупреждением или бросить исключение
        // Здесь бросаем исключение, так как запрошено "решение" в классическом смысле
        throw std::runtime_error("System is inconsistent (large residual)");
    }

    return x;
}


double EnumerationSolver::solveEnumeration() {
    const MatrixXd &A = problem._A;
    const VectorXd &b = problem._b;

    ColPivHouseholderQR<MatrixXd> qr(A.transpose());
    int rank = qr.rank();

    int m = (int)A.rows();
    int n = (int)A.cols();

    long num_of_additional_sign_equations = m - rank;
    if (num_of_additional_sign_equations < 0) {
        throw std::runtime_error("Strange: num_of_additional_sign_equations < 0");
    }

    std::vector<std::vector<int>> combinations = get_index_combinations(m, (int)num_of_additional_sign_equations);

    for (size_t comb_idx = 0; comb_idx < combinations.size(); ++comb_idx) {
        const auto &comb = combinations[comb_idx];
        std::vector<int> rows_to_keep = reverse_index_list(comb, m); // length should be == rank

//        // Диагностика
//        std::cerr << "comb #" << comb_idx << " comb.size()=" << comb.size()
//                  << " -> rows_to_keep.size()=" << rows_to_keep.size() << "\n";
//        for (size_t k = 0; k < rows_to_keep.size(); ++k) {
//            std::cerr << " rows_to_keep[" << k << "]=" << rows_to_keep[k];
//        }
//        std::cerr << "\n";

        Eigen::MatrixXd A_sub((Eigen::Index)rows_to_keep.size(), n);
        Eigen::VectorXd b_sub((Eigen::Index)rows_to_keep.size());

        for (size_t k = 0; k < rows_to_keep.size(); ++k) {
            int ridx = rows_to_keep[k];
            if (ridx < 0 || ridx >= m) throw std::runtime_error("row index out of range");
            A_sub.row((Eigen::Index)k) = A.row((Eigen::Index)ridx);
            b_sub((Eigen::Index)k) = b((Eigen::Index)ridx);
        }

        // Теперь размеры согласованы: A_sub.rows() == b_sub.size()
        try {
            Eigen::VectorXd x = solve_linear_system(A_sub, b_sub);
            std::cout << "Solution x (comb #" << comb_idx << "): " << x.transpose() << std::endl;
        } catch (const std::exception &e) {
//            std::cout << "Failed to solve for comb #" << comb_idx << ": " << e.what() << "\n";
            // либо continue, либо обрабатывать дальше
            continue;
        }
    }

    // вернуть что-то осмысленное; сейчас возвращаем 0.0
    return 0.0;
}

EnumerationSolver::~EnumerationSolver() = default;
