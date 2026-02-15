#pragma once
#include <Eigen/Dense>
#include <Eigen/QR>
#include <iostream>
#include <vector>
#include <limits>
#include <stdexcept>
#include "ProblemTypes/Canonical.h"

class Solver {
private:
    Canonical _problem;
    static constexpr double EPS = 1e-9;

    static std::tuple<Eigen::MatrixXd, Eigen::VectorXd, std::vector<int>> 
    removeDependentConstraints(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
        int m = A.rows();
        int n = A.cols();
        
        std::vector<int> independentRows;
        Eigen::MatrixXd selectedRows(0, n);
        
        for (int i = 0; i < m; ++i) {
            Eigen::RowVectorXd row = A.row(i);
            
            // Проверяем, линейно ли независима текущая строка от уже выбранных
            bool isIndependent = true;
            
            if (selectedRows.rows() > 0) {
                Eigen::MatrixXd combined(selectedRows.rows() + 1, n);
                combined.topRows(selectedRows.rows()) = selectedRows;
                combined.bottomRows(1) = row;
                
                Eigen::FullPivLU<Eigen::MatrixXd> lu(combined);
                lu.setThreshold(EPS);
                
                if (lu.rank() == selectedRows.rows()) {
                    isIndependent = false;
                }
            }
            
            if (isIndependent) {
                selectedRows.conservativeResize(selectedRows.rows() + 1, n);
                selectedRows.bottomRows(1) = row;
                independentRows.push_back(i);
            }
        }
        
        // Составляем новую матрицу и вектор
        Eigen::MatrixXd A_clean(independentRows.size(), n);
        Eigen::VectorXd b_clean(independentRows.size());
        
        for (size_t i = 0; i < independentRows.size(); ++i) {
            A_clean.row(i) = A.row(independentRows[i]);
            b_clean(i) = b(independentRows[i]);
        }
        
        return {A_clean, b_clean, independentRows};
    }

    static void make_b_nonneg(Eigen::MatrixXd& A, Eigen::VectorXd& b) {
        for (int i = 0; i < b.size(); ++i) {
            if (b(i) < -EPS) { 
                A.row(i) *= -1; 
                b(i) *= -1; 
            }
        }
    }

    static Canonical createAuxiliaryProblem(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
        Eigen::MatrixXd A_copy = A;
        Eigen::VectorXd b_copy = b;
        make_b_nonneg(A_copy, b_copy);
        
        const int m = A_copy.rows();
        const int n = A_copy.cols();

        Eigen::MatrixXd Aug(m, n + m);
        Aug.leftCols(n) = A_copy;
        Aug.rightCols(m).setIdentity();

        Eigen::VectorXd c_aug(n + m);
        c_aug.setZero();
        c_aug.tail(m).setOnes();

        std::vector<int> basisIndices(m);
        for (int i = 0; i < m; ++i) {
            basisIndices[i] = n + i;
        }

        Canonical aux_problem(Aug, b_copy, c_aug, basisIndices, true);
        aux_problem.SetOriginalVariablesCount(n);
        
        return aux_problem;
    }

    static Eigen::VectorXi complement(int n, const Eigen::VectorXi& N) {
        std::vector<char> inN(n, 0);
        for (int i = 0; i < N.size(); ++i) {
            int idx = N(i);
            if (idx < 0 || idx >= n) throw std::runtime_error("basis index out of range");
            inN[idx] = 1;
        }
        Eigen::VectorXi L(n - N.size());
        int t = 0;
        for (int j = 0; j < n; ++j) if (!inN[j]) L(t++) = j;
        return L;
    }

    static Eigen::MatrixXd basisMatrix(const Eigen::MatrixXd& A, const Eigen::VectorXi& N) {
        const int m = A.rows();
        Eigen::MatrixXd B(m, m);
        for (int t = 0; t < m; ++t) B.col(t) = A.col(N(t));
        return B;
    }

    static void computeBFS(const Eigen::MatrixXd& A,
                          const Eigen::VectorXd& b,
                          const Eigen::VectorXi& N,
                          Eigen::VectorXd& x,
                          Eigen::MatrixXd& Binv) {
        const int m = A.rows();
        Eigen::MatrixXd B = basisMatrix(A, N);
        Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
        if (!lu.isInvertible()) 
            throw std::runtime_error("Singular basis matrix");
        
        Binv = lu.inverse();
        Eigen::VectorXd xB = Binv * b;
        
        x.setZero(A.cols());
        for (int t = 0; t < m; ++t) x(N(t)) = xB(t);
    }

    std::string simplexIter(const Eigen::MatrixXd& A,
                           const Eigen::VectorXd& b,
                           const Eigen::VectorXd& c,
                           Eigen::VectorXi& N,
                           Eigen::MatrixXd& Binv,
                           bool isMaximization) {
        const int m = A.rows();
        const int n = A.cols();

        Eigen::RowVectorXd cB(m);
        for (int t = 0; t < m; ++t) cB(t) = c(N(t));
        Eigen::RowVectorXd yT = cB * Binv;

        Eigen::VectorXi L = complement(n, N);
        
        int enter = -1;
        
        if (isMaximization) {
            double max_d = -std::numeric_limits<double>::infinity();
            for (int t = 0; t < L.size(); ++t) {
                int j = L(t);
                double d = c(j) - (yT * A.col(j))(0,0);
                if (d > max_d + EPS) {
                    max_d = d;
                    enter = j;
                }
            }
            if (max_d <= EPS) return "optimal";
        } else {
            double min_d = std::numeric_limits<double>::infinity();
            for (int t = 0; t < L.size(); ++t) {
                int j = L(t);
                double d = c(j) - (yT * A.col(j))(0,0);
                if (d < min_d - EPS) {
                    min_d = d;
                    enter = j;
                }
            }
            if (min_d >= -EPS) return "optimal";
        }

        Eigen::VectorXd u = Binv * A.col(enter);
        Eigen::VectorXd xB = Binv * b;

        if ((u.array() <= EPS).all()) return "unbounded";

        double theta = std::numeric_limits<double>::infinity();
        int leave_pos = -1;
        
        for (int i = 0; i < m; ++i) {
            if (u(i) > EPS) {
                double r = xB(i) / u(i);
                if (r < theta - EPS) {
                    theta = r;
                    leave_pos = i;
                }
            }
        }

        if (leave_pos == -1) return "unbounded";

        N(leave_pos) = enter;

        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(m, m);
        for (int i = 0; i < m; ++i) {
            if (i != leave_pos) {
                F(i, leave_pos) = -u(i) / u(leave_pos);
            }
        }
        F(leave_pos, leave_pos) = 1.0 / u(leave_pos);
        
        Binv = F * Binv;

        return "iter";
    }

    static void replaceArtificialColumns(const Eigen::MatrixXd& A_aug,
                                        int n_orig,
                                        Eigen::VectorXi& N,
                                        Eigen::MatrixXd& Binv) {
        const int m = A_aug.rows();
        
        for (int pos = 0; pos < m; ++pos) {
            if (N(pos) < n_orig) continue;
            
            bool replaced = false;
            
            // Пытаемся найти небазисную исходную переменную для замены
            for (int cand = 0; cand < n_orig; ++cand) {
                // Проверяем, не базисная ли это переменная
                bool is_basic = false;
                for (int i = 0; i < m; ++i) {
                    if (N(i) == cand) { 
                        is_basic = true; 
                        break; 
                    }
                }
                if (is_basic) continue;
                
                Eigen::VectorXd u = Binv * A_aug.col(cand);
                
                if (std::abs(u(pos)) > EPS) {
                    N(pos) = cand;
                    
                    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(m, m);
                    for (int i = 0; i < m; ++i) {
                        if (i != pos) {
                            F(i, pos) = -u(i) / u(pos);
                        }
                    }
                    F(pos, pos) = 1.0 / u(pos);
                    
                    Binv = F * Binv;
                    replaced = true;
                    break;
                }
            }
            
            // Если не нашли замену, но искусственная переменная = 0,
            // просто удаляем эту строку и соответствующий базисный индекс
            if (!replaced) {
                Eigen::VectorXd x = Binv * A_aug.rightCols(m).col(pos);
                if (x.norm() < EPS) {
                    // Это лишнее ограничение - можно удалить
                    throw std::runtime_error("REDUNDANT_CONSTRAINT");
                } else {
                    throw std::runtime_error("Линейно зависимые ограничения");
                }
            }
        }
    }
    // =================================================================================

    static Eigen::VectorXi vectorToEigenVector(const std::vector<int>& vec) {
        Eigen::VectorXi result(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            result(i) = vec[i];
        }
        return result;
    }

    static std::vector<int> eigenVectorToVector(const Eigen::VectorXi& vec) {
        std::vector<int> result(vec.size());
        for (int i = 0; i < vec.size(); ++i) {
            result[i] = vec(i);
        }
        return result;
    }

public:
    Solver(const Canonical& problem) : _problem(problem) {}
    ~Solver() {}

    Eigen::VectorXd solve() {
        try {
            const std::vector<int>& initialBasis = _problem.GetBasisIndices();
            
            if (!initialBasis.empty()) {
                return solveWithBasis(_problem);
            } else {
                return twoPhaseSimplex(_problem);
            }
        } catch (const std::runtime_error& e) {
            std::string err = e.what();
            
            // ============== ОБРАБОТКА ЛИНЕЙНО ЗАВИСИМЫХ ОГРАНИЧЕНИЙ ==============
            if (err.find("REDUNDANT_CONSTRAINT") != std::string::npos ||
                err.find("Линейно зависимые") != std::string::npos) {
                
                std::cout << "⚠ Обнаружены линейно зависимые ограничения. Удаляем..." << std::endl;
                
                const Eigen::MatrixXd& A = _problem.GetConstraintsMatrix();
                const Eigen::VectorXd& b = _problem.GetRightHandSide();
                const Eigen::VectorXd& c = _problem.GetObjectiveCoefficients();
                const int n_orig = _problem.GetOriginalVariablesCount();
                const bool isMaximization = _problem.IsMaximization();
                
                // Удаляем зависимые ограничения
                auto [A_clean, b_clean, independentRows] = removeDependentConstraints(A, b);
                
                std::cout << "  Удалено " << (A.rows() - A_clean.rows()) << " зависимых ограничений" << std::endl;
                
                // Создаем новую задачу
                Canonical clean_problem(A_clean, b_clean, c, {}, !isMaximization);
                clean_problem.SetOriginalVariablesCount(n_orig);
                
                // Решаем заново
                return twoPhaseSimplex(clean_problem);
            }
            // =====================================================================
            
            throw;
        }
    }

private:
    Eigen::VectorXd twoPhaseSimplex(const Canonical& problem) {
        const Eigen::MatrixXd& A_orig = problem.GetConstraintsMatrix();
        const Eigen::VectorXd& b_orig = problem.GetRightHandSide();
        const Eigen::VectorXd& c_orig = problem.GetObjectiveCoefficients();
        const int n_orig = problem.GetOriginalVariablesCount();
        const bool isMaximization = problem.IsMaximization();
        
        // ФАЗА I
        Canonical aux_problem = createAuxiliaryProblem(A_orig, b_orig);
        Eigen::MatrixXd A_aug = aux_problem.GetConstraintsMatrix();
        Eigen::VectorXd b = aux_problem.GetRightHandSide();
        Eigen::VectorXd c = aux_problem.GetObjectiveCoefficients();
        std::vector<int> basisIndices = aux_problem.GetBasisIndices();

        const int m = A_aug.rows();
        const int n_aug = A_aug.cols();

        Eigen::VectorXi N = vectorToEigenVector(basisIndices);
        Eigen::VectorXd x(n_aug);
        Eigen::MatrixXd Binv;
        
        computeBFS(A_aug, b, N, x, Binv);

        const int MAX_ITER = 10000;
        int iteration = 0;
        
        while (iteration < MAX_ITER) {
            std::string status = simplexIter(A_aug, b, c, N, Binv, false);
            computeBFS(A_aug, b, N, x, Binv);
            
            double sum_artificial = 0.0;
            for (int i = 0; i < m; ++i) {
                sum_artificial += x(n_orig + i);
            }
            
            if (sum_artificial <= EPS) {
                break;
            }
            
            if (status == "optimal") {
                throw std::runtime_error("Задача не имеет допустимых решений");
            }
            
            if (status == "unbounded") {
                throw std::runtime_error("Вспомогательная задача неограничена");
            }
            
            iteration++;
        }

        // Заменяем искусственные переменные
        replaceArtificialColumns(A_aug, n_orig, N, Binv);
        
        // Формируем чистую матрицу
        Eigen::MatrixXd A_clean(m, n_orig);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n_orig; ++j) {
                A_clean(i, j) = A_aug(i, j);
            }
        }
        
        // Получаем чистый базис
        std::vector<int> cleanBasis;
        for (int i = 0; i < m; ++i) {
            if (N(i) >= n_orig) {
                throw std::runtime_error("Не удалось удалить искусственные переменные из базиса");
            }
            cleanBasis.push_back(N(i));
        }
        
        // ФАЗА II
        Canonical originalProblem(A_clean, b_orig, c_orig, cleanBasis, !isMaximization);
        originalProblem.SetOriginalVariablesCount(n_orig);
        
        return solveWithBasis(originalProblem);
    }

    Eigen::VectorXd solveWithBasis(const Canonical& problem) {
        const Eigen::MatrixXd& A = problem.GetConstraintsMatrix();
        const Eigen::VectorXd& b = problem.GetRightHandSide();
        const Eigen::VectorXd& c = problem.GetObjectiveCoefficients();
        const std::vector<int>& basisIndices = problem.GetBasisIndices();
        const int n_orig = problem.GetOriginalVariablesCount();
        const bool isMaximization = problem.IsMaximization();
        
        const int n = A.cols();
        const int m = A.rows();
        
        Eigen::VectorXi N = vectorToEigenVector(basisIndices);
        Eigen::VectorXd x(n);
        Eigen::MatrixXd Binv;
        
        computeBFS(A, b, N, x, Binv);
        
        
        const int MAX_ITER = 10000;
        int iteration = 0;
        
        while (iteration < MAX_ITER) {
            std::string status = simplexIter(A, b, c, N, Binv, isMaximization);
            
            if (status == "optimal") {
                computeBFS(A, b, N, x, Binv);
                
                Eigen::VectorXd x_opt(n_orig);
                for (int j = 0; j < n_orig; ++j) {
                    x_opt(j) = x(j);
                }
                return x_opt;
            }
            
            if (status == "unbounded") {
                throw std::runtime_error("Целевая функция неограничена");
            }
            
            computeBFS(A, b, N, x, Binv);
            iteration++;
        }
        
        throw std::runtime_error("Достигнут лимит итераций");
    }
};