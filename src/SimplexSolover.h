#pragma once

#include "ProblemTypes/Canonical.h"

class Solver
{
private:
    Canonical _problem;

    Canonical _createSupportProblem(Eigen::MatrixXd A, Eigen::VectorXd b){
        size_t m = b.rows();
        size_t n = A.cols();

        for(size_t i = 0; i < m; i++){
            if(b(i) < 0){
                b(i) *= -1;
                A(i,Eigen::all) *= -1;
            }
        }

        Eigen::VectorXd c(m);
        c.setOnes();


        Eigen::MatrixXd Aug(m, n + m);
        Aug.leftCols(n) = A;
        Aug.rightCols(m).setIdentity();

        std::cout << Aug << std::endl;

        return Canonical(A,b,c);
    }

public:
    Solver(Canonical problem) : _problem(problem){};
    ~Solver(){};

    void solve(){
        Eigen::VectorXi inp(2);
        inp << 2,3;
        solve(inp);
    }

    void solve(Eigen::VectorXi N_k){
        Eigen::MatrixXd A = _problem.getA();
        Eigen::VectorXd b = _problem.getb();
        Eigen::VectorXd c = _problem.getc();

        _createSupportProblem(A,b);

        size_t m = A.rows();
        size_t n = A.cols();

        // _getZeroSupportVector(A,b);

        std::cout << n - N_k.size() << std::endl;

        Eigen::VectorXi L_k(n - N_k.size());
        int t = 0;
        for (int j = 0; j < (int)n; ++j) {
            if (!(N_k.array() == j).any()) {
                L_k(t++) = j;
            }
        }
        

        Eigen::MatrixXd B = A(Eigen::all,N_k).inverse();
        Eigen::VectorXd y = c(N_k).transpose() * B;


        Eigen::VectorXd d_k(L_k.size());
        for(size_t j =0; j < L_k.size(); j++){      
            float d_ki = c[j] - y.transpose() * A.col(j);
            d_k(j) = d_ki;  
        }

        bool all_pos = (d_k.array() >= 0.0).all();
        if(all_pos){


            std::cout << "Опорный вектор:" << std::endl;
            
            Eigen::VectorXd solve(n);

            size_t k = 0;
            for(size_t i = 0; i < n; i++){

                if((N_k.array() == i).any()){
                    solve[i] = B(k,Eigen::all) * b;
                    k++;
                }
                else{
                    solve[i] = 0;
                }
            }
            std::cout << "(" << std::endl;
            std::cout << solve << std::endl;
            std::cout << ")" << std::endl;
        }
    }
};
