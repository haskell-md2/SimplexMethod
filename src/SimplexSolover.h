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

    static void make_b_nonneg(Eigen::MatrixXd& A, Eigen::VectorXd& b) {
        for (int i = 0; i < b.size(); ++i) {
            if (b(i) < 0) { A.row(i) *= -1; b(i) *= -1; }
        }
    }

    static Canonical createSupportProblem(Eigen::MatrixXd A, Eigen::VectorXd b) {
        make_b_nonneg(A, b);
        const int m = A.rows();
        const int n = A.cols();

        Eigen::MatrixXd Aug(m, n + m);
        Aug.leftCols(n) = A;
        Aug.rightCols(m).setIdentity();

        Eigen::VectorXd c_aug(n + m);
        c_aug.setZero();
        c_aug.tail(m).setOnes(); // min sum y

        return Canonical(Aug, b, c_aug);
    }

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

    // Построить B = A[:,N]
    static Eigen::MatrixXd basisMatrix(const Eigen::MatrixXd& A, const Eigen::VectorXi& N) {
        const int m = A.rows();
        Eigen::MatrixXd B(m, m);
        for (int t = 0; t < m; ++t) B.col(t) = A.col(N(t));
        return B;
    }

    // Пересчитать BFS x по текущему базису (небазис=0, базис=B^{-1}b)
    static void computePrimalBFS(const Eigen::MatrixXd& A,
                                const Eigen::VectorXd& b,
                                const Eigen::VectorXi& N,
                                Eigen::VectorXd& x_out) {
        Eigen::MatrixXd B = basisMatrix(A, N);
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(B);
        Eigen::VectorXd xB = lu.solve(b);

        x_out.setZero(A.cols());
        for (int t = 0; t < N.size(); ++t) x_out(N(t)) = xB(t);
    }

    // Один revised-simplex шаг (min). Обновляет базис. Возвращает: optimal/unbounded/iter
    std::string simplexIterMin(const Eigen::MatrixXd& A,
                                        const Eigen::VectorXd& b,
                                        const Eigen::VectorXd& c,
                                        Eigen::VectorXi& N,
                                        Eigen::VectorXd& x) {
        const int m = A.rows();
        const int n = A.cols();

        Eigen::VectorXi L = complement(n, N);

        Eigen::MatrixXd B = basisMatrix(A, N);
        Eigen::MatrixXd Binv = B.inverse();

        Eigen::VectorXd xB = Binv * b;
        x.setZero(n);
        for (int t = 0; t < m; ++t) x(N(t)) = xB(t);

        Eigen::RowVectorXd cB(m);
        for (int t = 0; t < m; ++t) cB(t) = c(N(t));
        Eigen::RowVectorXd yT = cB * Binv;

        // entering: любой j с d_j < 0, возьмём самый отрицательный
        double best_d = 0.0;
        int enter = -1;
        for (int t = 0; t < L.size(); ++t) {
            int j = L(t);
            double d = c(j) - (yT * A.col(j))(0,0);    // d_j = c_j - y^T a_j
            if (d < best_d - 1e-15) { best_d = d; enter = j; }
        }
        if (enter == -1) return "optimal";

        Eigen::VectorXd u = Binv * A.col(enter);       // u = B^{-1} a_enter
        if ((u.array() <= EPS).all()) return "unbounded";

        double theta = std::numeric_limits<double>::infinity();
        int leave_pos = -1;
        for (int i = 0; i < m; ++i) {
            if (u(i) > EPS) {
                double r = xB(i) / u(i);
                if (r < theta) { theta = r; leave_pos = i; }
            }
        }
        if (leave_pos == -1 || !std::isfinite(theta)) return "unbounded";

        N(leave_pos) = enter;
        return "iter";
    }

    // “Выпихнуть” искусственные из базиса (если остались) после Phase I
    // A_aug = [A|I], n_orig = A.cols(), искусственные индексы: n_orig..n_orig+m-1
    static void pivotOutArtificial(const Eigen::MatrixXd& A_aug,
                                  const Eigen::VectorXd& b,
                                  int n_orig,
                                  Eigen::VectorXi& N) {
        const int m = A_aug.rows();
        const int n_aug = A_aug.cols();
        (void)n_aug;

        // пересчёт x, чтобы знать, где вырожденность (обычно искусственные в базе будут 0)
        Eigen::VectorXd x_aug;
        computePrimalBFS(A_aug, b, N, x_aug);

        // Для каждой базисной искусственной переменной пробуем найти входящий "настоящий" столбец,
        // чтобы пивот был возможен (ненулевой pivot в соответствующей строке).
        for (int pos = 0; pos < m; ++pos) {
            int var = N(pos);
            if (var < n_orig) continue; // уже не искусственная

            Eigen::MatrixXd B = basisMatrix(A_aug, N);
            Eigen::PartialPivLU<Eigen::MatrixXd> lu(B);

            bool replaced = false;
            for (int cand = 0; cand < n_orig; ++cand) {
                // cand должен быть небазисным
                if ((N.array() == cand).any()) continue;

                Eigen::VectorXd u = lu.solve(A_aug.col(cand));
                if (std::abs(u(pos)) > 1e-9) {
                    // делаем pivot: заменяем искусственную на cand
                    N(pos) = cand;
                    replaced = true;
                    break;
                }
            }

            // Если не смогли заменить, то соответствующее ограничение линейно зависимо/лишнее:
            // искусственная переменная останется базисной с нулём.
            // Для Phase II это означает, что базис по исходным переменным может не существовать без удаления строки.
            // Здесь просто оставим как есть (но предупредим).
            if (!replaced) {
                // проверим, что она действительно 0 (иначе это противоречит obj=0)
                if (x_aug(var) > 1e-8) {
                    throw std::runtime_error("Artificial basic variable is positive after Phase I optimal=0.");
                }
            }
        }
    }

public:
    Solver(Canonical problem) : _problem(problem) {}
    ~Solver() {}

    // Возвращает x* для исходной задачи (длины n_orig)
    Eigen::VectorXd solve() {
        // ---------- Вспомогательная задача ----------
        Canonical sup = createSupportProblem(_problem.getA(), _problem.getb());
        Eigen::MatrixXd A_aug = sup.getA();
        Eigen::VectorXd b = sup.getb();
        Eigen::VectorXd c_aug = sup.getc();

        const int m = A_aug.rows();
        const int n_aug = A_aug.cols();
        const int n_orig = _problem.getA().cols();

        // начальный базис: искусственные y
        Eigen::VectorXi N(m);
        for (int i = 0; i < m; ++i) N(i) = n_orig + i;

        Eigen::VectorXd x_aug(n_aug);

        for (int it = 0; it < 20'000; ++it) {
            std::string st = simplexIterMin(A_aug, b, c_aug, N, x_aug);
            if (st == "optimal") break;
            if (st == "unbounded") throw std::runtime_error("Phase I unbounded (unexpected).");
        }

        // проверка infeasible
        computePrimalBFS(A_aug, b, N, x_aug);
        double w = c_aug.dot(x_aug); // w = sum y
        if (w > 1e-8) {
            throw std::runtime_error("Infeasible original problem (Phase I optimum > 0).");
        }

        // выпихиваем искусственные из базиса, если они ещё там
        pivotOutArtificial(A_aug, b, n_orig, N);

        // ---------- Build Phase II start ----------
        // Используем тот же базис N (но уже должен быть по возможности < n_orig),
        // и запускаем simplex на исходной A, c.
        Eigen::MatrixXd A = _problem.getA();
        Eigen::VectorXd c = _problem.getc();
        make_b_nonneg(A, b); // важно: b в sup уже был >=0, но A тут исходная; привести согласованно

        // Если в N всё ещё есть искусственные индексы >= n_orig, то без удаления строк базис в A не построить.
        if ((N.array() >= n_orig).any()) {
            throw std::runtime_error("Cannot start Phase II: artificial vars remain in basis. Need to drop redundant rows.");
        }

        Eigen::VectorXd x(n_orig);
        for (int it = 0; it < 50'000; ++it) {
            std::string st = simplexIterMin(A, b, c, N, x);
            if (st == "optimal") {
                // x уже длины n_orig и является BFS/оптимумом
                return x;
            }
            if (st == "unbounded") {
                throw std::runtime_error("Original problem unbounded.");
            }
        }

        throw std::runtime_error("Iteration limit in Phase II.");
    }
};
