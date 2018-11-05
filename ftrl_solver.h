#ifndef FTRL_SOLVER_H
#define FTRL_SOLVER_H

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <functional>
#include <iomanip>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <memory.h>
#include <map>
#include "mpi.h"

#define DEFAULT_ALPHA 0.15
#define DEFAULT_BETA 1.
#define DEFAULT_L1 1.
#define DEFAULT_L2 1.
#define MAX_EXP_NUM 50.

double safe_exp(double x) {
    double max_exp = static_cast<double>(MAX_EXP_NUM);
    return std::exp(std::max(std::min(x, max_exp), -max_exp));
}

double sigmoid(double x) {
    double one = 1.;
    return one / (one + safe_exp(-x));
}

class FtrlSolver {
public:
    FtrlSolver();

    virtual ~FtrlSolver();
 
    virtual bool Initialize(double alpha, double beta, double l1, double l2, size_t n);

    virtual std::vector<double> UpdateBatch(const std::vector<std::vector<std::pair<size_t, double> > >& x, std::vector<double>& y);

    virtual double Predict(const std::vector<std::pair<size_t, double> >& x);

public:
    double alpha() { return alpha_; }
    double beta() { return beta_; }
    double l1() { return l1_; }
    double l2() { return l2_; }
    size_t feat_num() { return feat_num_; }

    double GetWeight(size_t idx);
    int GlobalUpdate();
    int Push(int rank);
    int Pull();

public:
    double alpha_;
    double beta_;
    double l1_;
    double l2_;
    size_t feat_num_;

    double *z_;
    double *n_;
    double *g_sum_;
    int *g_num_;

    bool init_;
};

FtrlSolver::FtrlSolver(): alpha_(0), beta_(0), l1_(0), l2_(0), feat_num_(0), n_(NULL), z_(NULL), g_sum_(NULL), g_num_(NULL), init_(false) {}

FtrlSolver::~FtrlSolver() {
    if (n_) {
        delete [] n_;
    }
    if (z_) {
        delete [] z_;
    }
    if (g_sum_) {
        delete [] g_sum_;
    }
    if (g_num_) {
        delete [] g_num_;
    }
}

bool FtrlSolver::Initialize(double alpha, double beta, double l1, double l2, size_t n) {
    alpha_ = alpha;
    beta_ = beta;
    l1_ = l1;
    l2_ = l2;
    feat_num_ = n;

    n_ = new double[feat_num_];
    z_ = new double[feat_num_];

    g_sum_ = new double[feat_num_];
    g_num_ = new int[feat_num_];

    memset(n_, 0, feat_num_);
    memset(z_, 0, feat_num_);
    memset(g_sum_, 0, feat_num_);
    memset(g_num_, 0, feat_num_);

    //MPI_Data_type old_type[] = {MPI_INT, MPI_DOUBLE};
    //int blocklen[2] = {1, 1};
    //MPI_Aint offsets[2];
    //offsets[0] = offsetof(Node, id);
    //offsets[1] = offsetof(Node, g_i);
    //MPI_Data_type MPI_Node;
    //MPI_Type_create_struct(2, blocklen, offsets, old_type, &MPI_Node);
    //MPI_Type_commit(&MPI_Node);

    init_ = true;
    return init_;
}

double FtrlSolver::GetWeight(size_t idx) {
    if (idx > feat_num_) {
        return 0;
    }

    auto sign = [&](double z) {
        if (z < 0)
            return -1.;
        else
            return 1.;
    };

    auto less_equal = [&](double v1, double v2) {
	if (std::fabs(v1 - v2) < std::numeric_limits<double>::epsilon()) {
            return true;
        }
        return v1 < v2;
    };

    double val = 0.;
    if (less_equal(sign(z_[idx]) * z_[idx] , l1_)) {
        val = 0.;
    } else {
        val = (sign(z_[idx]) * l1_ - z_[idx]) / ((beta_ + sqrt(n_[idx])) / alpha_ + l2_);
    }

    return val;
}

std::vector<double> FtrlSolver::UpdateBatch(const std::vector<std::vector<std::pair<size_t, double> > >& batch_x, std::vector<double>& batch_y) {
    if (!init_) 
        return std::vector<double>();

    std::vector<double> pred;

    memset(g_sum_, 0, feat_num_);
    memset(g_num_, 0, feat_num_);

    for (int index = 0; index < batch_x.size(); ++index) {
        std::vector<std::pair<size_t, double> > x = batch_x[index];
        double y = batch_y[index];
        double wTx = 0.;
        std::vector<size_t> id_v;
        for (auto& item : x) {
            size_t idx = item.first;
            if (idx > feat_num_)
                continue;

            double val = GetWeight(idx);
            wTx += val * item.second;
            id_v.push_back(idx);
        }
        double p = sigmoid(wTx);
        double grad = p - y;
        pred.push_back(p);
        //std::cout << p << " ";
        for (auto id : id_v) {
            g_sum_[id] += grad;
            g_num_[id] += 1;
        }
    }

    return pred;
}

double FtrlSolver::Predict(const std::vector<std::pair<size_t, double> >& x) {
    if (!init_) 
        return 0;

    double wTx = 0.;
    for (auto& item : x) {
        size_t idx = item.first;
        double val = GetWeight(idx);
        wTx += val * item.second;
    }

    double pred = sigmoid(wTx);
    return pred;
}

int FtrlSolver::Push(int rank) {
    //std::cout << "rank " << rank << " push to master" << std::endl;
    MPI_Send(g_sum_, feat_num_, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
    MPI_Send(g_num_, feat_num_, MPI_INT, 0, rank*2, MPI_COMM_WORLD);
    return 0;
}

int FtrlSolver::Pull() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Bcast(z_, feat_num_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_, feat_num_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //std::cout << "rank " << rank << " pull from master" << std::endl;
    //std::cout << "rank " << rank << << " " << z_[0] << " " << n_[0] << std::endl;
    return 0;
}

int FtrlSolver::GlobalUpdate() {
    int n_procs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    double global_g_sum[feat_num_];
    int global_g_num[feat_num_];
    memset(g_sum_, 0, feat_num_);
    memset(g_num_, 0, feat_num_);
    memset(global_g_sum, 0, feat_num_);
    memset(global_g_num, 0, feat_num_);
    for (int i = 1; i < n_procs; ++i) {
        //std::cout << "master recv g_sum from rank: " << i << std::endl;
        MPI_Recv(global_g_sum, feat_num_, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //std::cout << "master recv g_num from rank: " << i << std::endl;
        MPI_Recv(global_g_num, feat_num_, MPI_INT, i, i*2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 0; j < feat_num_; ++j) {
            g_sum_[j] += global_g_sum[j];
            g_num_[j] += global_g_num[j];
        }
    }
    //std::cout << "master begin update" << std::endl;
    for (int i = 0; i < feat_num_; ++i) {
        if (g_num_[i] > 0) {
            double grad_i = g_sum_[i] / g_num_[i];
            double w_i = GetWeight(i);
            double sigma = (sqrt(n_[i] + grad_i * grad_i) - sqrt(n_[i])) / alpha_;
            z_[i] += grad_i - sigma * w_i;
            n_[i] += grad_i * grad_i;
        }
    }
    //std::cout << "master finish update" << std::endl;
    return 0;
}


#endif
