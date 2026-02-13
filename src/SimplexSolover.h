#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <limits>
#include <stdexcept>

#include "ProblemTypes/Canonical.h"

class Solver {
private:
    Canonical _problem;
    static constexpr double EPS = 1e-12;

    // Приведение b к неотрицательному виду (умножение строк на -1)
    static void make_b_nonneg(Eigen::MatrixXd& A, Eigen::VectorXd& b) {
        for (int i = 0; i < b.size(); ++i) {
            if (b(i) < 0) { 
                A.row(i) *= -1; 
                b(i) *= -1; 
            }
        }
    }

    // Построение расширенной задачи (5.11) с целевой функцией min sum(y)
    static Canonical createAuxiliaryProblem(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
        Eigen::MatrixXd A_copy = A;
        Eigen::VectorXd b_copy = b;
        make_b_nonneg(A_copy, b_copy);
        
        const int m = A_copy.rows();
        const int n = A_copy.cols();

        // [A | I]
        Eigen::MatrixXd Aug(m, n + m);
        Aug.leftCols(n) = A_copy;
        Aug.rightCols(m).setIdentity();

        // Целевая функция: min sum(y)
        Eigen::VectorXd c_aug(n + m);
        c_aug.setZero();
        c_aug.tail(m).setOnes();

        return Canonical(Aug, b_copy, c_aug);
    }

    // Вычисление L = N \ N_k
    static Eigen::VectorXi complement(int n, const Eigen::VectorXi& N) {
        std::vector<char> inN(n, 0);
        for (int i = 0; i < N.size(); ++i) {
            int idx = N(i);
            if (idx < 0 || idx >= n) throw std::runtime_error("basis index out of range");
            if (inN[idx]) throw std::runtime_error("duplicate index in basis");
            inN[idx] = 1;
        }
        Eigen::VectorXi L(n - N.size());
        int t = 0;
        for (int j = 0; j < n; ++j) if (!inN[j]) L(t++) = j;
        return L;
    }

    // Построение базисной матрицы B = A[:, N]
    static Eigen::MatrixXd basisMatrix(const Eigen::MatrixXd& A, const Eigen::VectorXi& N) {
        const int m = A.rows();
        Eigen::MatrixXd B(m, m);
        for (int t = 0; t < m; ++t) B.col(t) = A.col(N(t));
        return B;
    }

    // Вычисление B^{-1} и базисного решения
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

    // Одна итерация симплекс-метода
    std::string simplexIter(const Eigen::MatrixXd& A,
                           const Eigen::VectorXd& b,
                           const Eigen::VectorXd& c,
                           Eigen::VectorXi& N,
                           Eigen::VectorXd& x,
                           Eigen::MatrixXd& Binv) {
        const int m = A.rows();
        const int n = A.cols();

        // Шаг 1: Вычисляем y = c_B^T * B^{-1}
        Eigen::RowVectorXd cB(m);
        for (int t = 0; t < m; ++t) cB(t) = c(N(t));
        Eigen::RowVectorXd yT = cB * Binv;

        // Шаг 2: Вычисляем оценки d_j для небазисных переменных
        Eigen::VectorXi L = complement(n, N);
        
        double best_d = 0.0;
        int enter = -1;
        
        for (int t = 0; t < L.size(); ++t) {
            int j = L(t);
            double d = c(j) - (yT * A.col(j))(0,0);
            if (d < best_d - 1e-15) {
                best_d = d;
                enter = j;
            }
        }

        // Если все d_j >= 0 - достигнут оптимум
        if (enter == -1) return "optimal";

        Eigen::VectorXd u = Binv * A.col(enter);

        if ((u.array() <= EPS).all()) return "unbounded";


        Eigen::VectorXd xB = Binv * b;
        
        double theta = std::numeric_limits<double>::infinity();
        int leave_pos = -1;
        
        for (int i = 0; i < m; ++i) {
            if (u(i) > EPS) {
                double r = xB(i) / u(i);
                if (r < theta - 1e-15) {
                    theta = r;
                    leave_pos = i;
                }
            }
        }

        if (leave_pos == -1) return "unbounded";

        //Обновляем базис
        int leave_var = N(leave_pos);
        N(leave_pos) = enter;

        // Обновляем B^{-1} по формуле (5.8) из учебника
        // Строим матрицу G = B^{-1} * A[:, N_{k+1}]
        Eigen::MatrixXd G = Eigen::MatrixXd::Identity(m, m);
        G.col(leave_pos) = u;  // вектор u становится на место выходящего столбца
        
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(m, m);
        for (int i = 0; i < m; ++i) {
            if (i != leave_pos) {
                F(i, leave_pos) = -u(i) / u(leave_pos);
            }
        }
        F(leave_pos, leave_pos) = 1.0 / u(leave_pos);
        
        // Обновляем B^{-1}
        Binv = F * Binv;

        return "iter";
    }

    // Замена искусственных столбцов в базисе (только для вырожденного случая)
    static void replaceArtificialColumns(const Eigen::MatrixXd& A_aug,
                                        const Eigen::VectorXd& b,
                                        int n_orig,
                                        Eigen::VectorXi& N,
                                        Eigen::MatrixXd& Binv,
                                        const Eigen::VectorXd& x) {
        const int m = A_aug.rows();
        
        for (int pos = 0; pos < m; ++pos) {
            int var = N(pos);
            if (var < n_orig) continue;  // уже исходная переменная
            
            // Пытаемся найти небазисный исходный столбец для замены
            bool replaced = false;
            for (int cand = 0; cand < n_orig; ++cand) {
                // cand должен быть небазисным
                bool is_basic = false;
                for (int i = 0; i < m; ++i) {
                    if (N(i) == cand) { is_basic = true; break; }
                }
                if (is_basic) continue;
                
                // Вычисляем u = B^{-1} * a_{cand}
                Eigen::VectorXd u = Binv * A_aug.col(cand);
                
                // Проверяем, можно ли использовать cand для замены
                if (std::abs(u(pos)) > 1e-9) {
                    // Заменяем искусственную на cand
                    N(pos) = cand;
                    
                    // Обновляем B^{-1} как при обычной итерации
                    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(m, m);
                    G.col(pos) = u;
                    
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
            
            // Если не удалось заменить, проверяем что искусственная = 0
            if (!replaced) {
                if (std::abs(x(var)) > 1e-8) {
                    throw std::runtime_error("Artificial variable positive and cannot be replaced");
                }
                // Иначе - избыточное ограничение, оставляем в базисе
            }
        }
    }

public:
    Solver(Canonical problem) : _problem(problem) {}
    ~Solver() {}

    Eigen::VectorXd solve() {
        const Eigen::MatrixXd& A_orig = _problem.getA();
        const Eigen::VectorXd& b_orig = _problem.getb();
        const Eigen::VectorXd& c_orig = _problem.getc();

        // ------------------------------------------------------------
        // Строим вспомогательную задачу (5.11) с min sum(y)
        // ------------------------------------------------------------
        Canonical aux_problem = createAuxiliaryProblem(A_orig, b_orig);
        Eigen::MatrixXd A_aug = aux_problem.getA();
        Eigen::VectorXd b = aux_problem.getb();
        Eigen::VectorXd c = aux_problem.getc();  // min sum(y)

        const int m = A_aug.rows();
        const int n_aug = A_aug.cols();
        const int n_orig = A_orig.cols();

        // ------------------------------------------------------------
        // Начальный опорный вектор: x = 0, y = b >= 0
        // Базис: все искусственные переменные
        // ------------------------------------------------------------
        Eigen::VectorXi N(m);
        for (int i = 0; i < m; ++i) N(i) = n_orig + i;
        
        Eigen::VectorXd x(n_aug);
        Eigen::MatrixXd Binv;
        
        // Вычисляем начальные B, B^{-1}, x
        computeBFS(A_aug, b, N, x, Binv);

        // ------------------------------------------------------------
        // Решаем вспомогательную задачу симплекс-методом
        // ------------------------------------------------------------
        bool artificials_zero = false;
        int iteration = 0;
        const int MAX_ITER = 100000;
        
        while (iteration < MAX_ITER) {
            std::string status = simplexIter(A_aug, b, c, N, x, Binv);
            
            // Пересчитываем BFS после итерации
            computeBFS(A_aug, b, N, x, Binv);
            
            // Проверяем, обнулились ли искусственные переменные
            if (!artificials_zero) {
                double sum_y = 0.0;
                for (int i = 0; i < m; ++i) sum_y += x(n_orig + i);
                
                if (sum_y <= 1e-8) {
                    artificials_zero = true;
                    
                    // Если x - вырожденный опорный вектор и в базисе
                    // остались искусственные столбцы - заменяем их
                    replaceArtificialColumns(A_aug, b, n_orig, N, Binv, x);
                    
                    // Пересчитываем BFS после замен
                    computeBFS(A_aug, b, N, x, Binv);
                }
            }
            
            if (status == "optimal") {
                // Проверяем, все ли искусственные равны 0
                double final_sum_y = 0.0;
                for (int i = 0; i < m; ++i) final_sum_y += x(n_orig + i);
                
                if (final_sum_y > 1e-8) {
                    throw std::runtime_error("Задача неразрешима");
                }
                
                break;
            }
            
            if (status == "unbounded") {
                // Вспомогательная задача не может быть неограниченной
                throw std::runtime_error("Вспомогательная задача неограниченна");
            }
            
            iteration++;
        }

        // ------------------------------------------------------------
        // Решаем исходную задачу с целевой функцией c_orig
        // ------------------------------------------------------------
        Eigen::MatrixXd A = _problem.getA();
        Eigen::VectorXd c_orig_copy = c_orig;
        
        // Приводим b к неотрицательному виду (согласованно с A)
        make_b_nonneg(A, b);
        
        // Убеждаемся, что в базисе нет искусственных переменных
        for (int i = 0; i < m; ++i) {
            if (N(i) >= n_orig) {
                throw std::runtime_error("Остались эл-ты искусственного базиса");
            }
        }
        
        // Вычисляем BFS для исходной задачи с текущим базисом
        computeBFS(A, b, N, x, Binv);
        
        // Симплекс-метод для исходной задачи
        iteration = 0;
        while (iteration < MAX_ITER) {
            std::string status = simplexIter(A, b, c_orig_copy, N, x, Binv);
            
            if (status == "optimal") {
                // Извлекаем решение исходных переменных
                Eigen::VectorXd x_opt(n_orig);
                for (int j = 0; j < n_orig; ++j) x_opt(j) = x(j);
                return x_opt;
            }
            
            if (status == "unbounded") {
                throw std::runtime_error("Функция неограниченна");
            }
            
            computeBFS(A, b, N, x, Binv);
            iteration++;
        }
        
        throw std::runtime_error("Зацикливание!");
    }
};