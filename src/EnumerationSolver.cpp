#include "EnumerationSolver.h"
#include <Eigen/Dense>
#include <Eigen/QR>
#include <iostream>
#include <vector>
#include <limits>

using namespace Eigen;

EnumerationSolver::EnumerationSolver(const Canonical &problem) : problem(problem) {}

EnumerationSolver::~EnumerationSolver() = default;

void combine_indices(int n, int k, int start, std::vector<int>& current, 
                    std::vector<std::vector<int>>& result) {
    if (static_cast<int>(current.size()) == k) {
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
                                       const std::vector<int> &cols) {
    const MatrixXd &A = problem.GetConstraintsMatrix();
    const VectorXd &b = problem.GetRightHandSide();

    // Проверка на NaN/Inf
    bool okFinite = true;
    for (int i = 0; i < xB.size(); ++i) {
        if (!std::isfinite(xB(i))) { 
            okFinite = false; 
            break; 
        }
    }
    if (!okFinite) return true;

    // Восстановление полного вектора
    Eigen::VectorXd x_full = Eigen::VectorXd::Zero(A_num_cols);
    for (int j = 0; j < rank; ++j) {
        x_full(cols[j]) = xB(j);
    }

    // Проверка допустимости (неотрицательность)

    if (x_full.minCoeff() < -tol_feas) return true;

    // Проверка выполнения всех уравнений A*x = b (по всей системе)
    double rel_res = (A * x_full - b).norm();
    if (rel_res > tol_res) return true; // САМАЯ СОМНИТЕЛЬНАЯ ПРОВЕРКА

    // Дедупликация (если уже есть похожая вершина, пропускаем)
    bool duplicate = false;
    for (const auto &v : found_vertices) {
        if ((v - x_full).norm() < tol_dup) {
            return true;
        }
    }

    found_vertices.push_back(x_full);
    return false;
}

double EnumerationSolver::findSolution() {
    const VectorXd &c = problem.GetObjectiveCoefficients();
    bool isMaximization = problem.IsMaximization();
    
    double best = isMaximization ? 
        -std::numeric_limits<double>::infinity() : 
         std::numeric_limits<double>::infinity();

    for (const auto &x_full : found_vertices) {
        double val = c.dot(x_full);
        
        if (isMaximization) {
            if (val > best) best = val;
        } else {
            if (val < best) best = val;
        }
        
//        std::cout << "Feasible vertex: x = " << x_full.transpose() << "  f = " << val << std::endl;
    }
    
    return best;
}

void EnumerationSolver::solveEnumeration() {
    const MatrixXd &A = problem.GetConstraintsMatrix();
    const VectorXd &b = problem.GetRightHandSide();

    ColPivHouseholderQR<MatrixXd> qr(A.transpose());  // транспонируем
    int rank = qr.rank();
    int A_num_cols = static_cast<int>(A.cols());

    VectorXi perm = qr.colsPermutation().indices();
    std::vector<int> independent_rows;
    for (int i = 0; i < rank; ++i) {
        independent_rows.push_back(perm(i));
    }

    MatrixXd B(static_cast<Eigen::Index>(rank), A_num_cols);
    VectorXd b_sub(static_cast<Eigen::Index>(rank));
    for (int i = 0; i < rank; ++i) {
        B.row(i) = A.row(independent_rows[i]); // новая матрица с независимыми строками
        b_sub(i) = b(independent_rows[i]);     // соотв B вектор
    }

    std::vector<std::vector<int>> col_combs = get_index_combinations(A_num_cols, rank);

    for (size_t comb_idx = 0; comb_idx < col_combs.size(); ++comb_idx) {
        const auto &cols = col_combs[comb_idx];

        MatrixXd B_cols(static_cast<Eigen::Index>(rank), static_cast<Eigen::Index>(rank));
        for (int j = 0; j < rank; ++j) {
            B_cols.col(j) = B.col(cols[j]);
        }

        // 1) проверим ранг базисной матрицы
        Eigen::FullPivLU<MatrixXd> fplu(B_cols);
        if (fplu.rank() < rank) continue; // вырожденный базис

        // 2) решаем быстро (PartialPivLU)
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(B_cols);
        Eigen::VectorXd xB = lu.solve(b_sub);

        handleSolution(xB, rank, A_num_cols, cols); // добавляет вершину в found_vertices
    }
}


Eigen::VectorXd EnumerationSolver::getBestVertex() {
    if (found_vertices.empty()) {
        throw std::runtime_error("Не найдено допустимых вершин");
    }
    
    const VectorXd &c = problem.GetObjectiveCoefficients();
    bool isMaximization = problem.IsMaximization();
    int n_orig = problem.GetOriginalVariablesCount();
    
    Eigen::VectorXd best_vertex;
    double best_value = isMaximization ? 
        -std::numeric_limits<double>::infinity() : 
         std::numeric_limits<double>::infinity();
    
    for (const auto &vertex : found_vertices) {
        double val = c.dot(vertex);
        
        if ((isMaximization && val > best_value) || 
            (!isMaximization && val < best_value)) {
            best_value = val;
            best_vertex = vertex;
        }
    }
    
    // Просто возвращаем первые n_orig переменных без преобразования
    // Преобразование из симметричной в исходную форму нужно делать
    // на уровне выше, здесь просто берем head(n_orig)
    return best_vertex.head(n_orig);
}

