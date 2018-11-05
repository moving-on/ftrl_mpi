#ifndef FTRL_TRAIN_H
#define FTRL_TRAIN_H

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <file_parser.h>
#include <ftrl_solver.h>
#include <iostream>

#define MIN_SIGMOID (10e-15)
#define MAX_SIGMOID (1. - 10e-15)
const int MASTER_ID = 0;

size_t read_problem_info(
	const char* train_file,
	size_t& line_cnt, int rank);

template<class Func>
double evaluate_file(const char* test_file, const Func& func_predict);

double calc_loss(double y, double pred) {
	double max_sigmoid = static_cast<double>(MAX_SIGMOID);
	double min_sigmoid = static_cast<double>(MIN_SIGMOID);
	double one = 1.;
	pred = std::max(std::min(pred, max_sigmoid), min_sigmoid);
	double loss = y > 0 ? -log(pred) : -log(std::max(one - pred, min_sigmoid));
	return loss;
}

class FtrlTrainer {
public:
    FtrlTrainer();

    virtual ~FtrlTrainer();

    bool Initialize(size_t epoch, int feat_num, double alpha, double beta, double l1, double l2);

    int Train(double alpha, double beta, double l1, double l2, size_t feat_num, size_t line_cnt,
            const char* model_file, const char* train_file, 
            const char* test_file, uint32_t mini_batch);

protected:
    int TrainImplBatch(const char* model_file, const char* train_file, size_t line_cnt, const char* test_file, uint32_t mini_batch);

private:
    size_t epoch_;
    bool init_;
    FtrlSolver solver_;
};

FtrlTrainer::FtrlTrainer(): epoch_(0) {}

FtrlTrainer::~FtrlTrainer() {}

bool FtrlTrainer::Initialize(size_t epoch, int feat_num, double alpha, double beta, double l1, double l2) {
    epoch_ = epoch;
    if (!solver_.Initialize(alpha, beta, l1, l2, feat_num)) {
        return false;
    }
    init_ = true;
    return init_;
}

size_t read_problem_info(const char* train_file, size_t& line_cnt, int rank) {
    size_t feat_num = 0;
    line_cnt = 0;

    FileParser parser;

    auto read_problem_worker = [&]() {
        std::vector<std::pair<size_t, double> > x;
        double y;
        while (1) {
            if (!parser.ReadSample(y, x))
                break;
            for (auto item : x) {
                if (item.first + 1 > feat_num)
                    feat_num = item.first + 1;
            }
            line_cnt += 1;
        }
    };

    parser.OpenFile(train_file);
    fprintf(stdout, "rank %d: loading...\n", rank);
    fflush(stdout);
    read_problem_worker();
    parser.CloseFile();

    fprintf(stdout, "\rinstances=[%zu] features=[%zu]\n", line_cnt, feat_num);

    return feat_num;
}

int FtrlTrainer::Train(double alpha, double beta, double l1, double l2, size_t feat_num, size_t line_cnt,
        const char* model_file, const char* train_file, const char* test_file, uint32_t mini_batch) {

    if (!init_)
        return false;
    int rank = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    return TrainImplBatch(model_file, train_file, line_cnt, test_file, mini_batch);
}

int FtrlTrainer::TrainImplBatch(const char* model_file, const char* train_file, size_t line_cnt, 
        const char* test_file, uint32_t mini_batch) {
    if (!init_) 
        return -1;

    fprintf(stdout,
            "params={alpha:%.2f, beta:%.2f, l1:%.2f, l2:%.2f, epoch:%zu}\n",
            solver_.alpha(), solver_.beta(), solver_.l1(), solver_.l2(), epoch_);

    auto predict_func = [&] (const std::vector<std::pair<size_t, double> >& x) {
        return solver_.Predict(x);
    };

    int rank, n_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (size_t iter = 0; iter < epoch_; ++iter) {
        FileParser file_parser;
        file_parser.OpenFile(train_file);
        std::vector<std::pair<size_t, double> > x;
        double y;

        double loss = 0;
        size_t cur_cnt = 0;
        for (int batch = 0; batch < 140; ++batch) {
            size_t local_cnt = 0;
            std::vector<std::vector<std::pair<size_t, double> > > batch_x;
            std::vector<double> batch_y;
            if (rank != MASTER_ID) {
                for (uint32_t i = 0; i < mini_batch; ++i) {
                    if (!file_parser.ReadSample(y, x))
                        break;
                    batch_x.push_back(x);
                    batch_y.push_back(y);
                    cur_cnt += 1;
                    local_cnt += 1;
                }
                std::vector<double> pred = solver_.UpdateBatch(batch_x, batch_y);
                for (int i = 0; i < pred.size(); ++i) { 
                    loss += calc_loss(batch_y[i], pred[i]);
                }
                //std::cout << "rank=" << rank << " :loss" << loss << std::endl;
            }
            
            // push or pull
            if (rank == MASTER_ID) {
                solver_.GlobalUpdate();
            } else {
                solver_.Push(rank);
            }
            
            /*
            int is_local_over = 0;
            int is_all_over = 0;
            if (rank != MASTER_ID) {
                if (local_cnt < mini_batch) {
                    is_local_over = 1;
                }
            } else {
                is_local_over = 1;
            }
            MPI_Reduce(&is_local_over, &is_all_over, n_procs, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == MASTER_ID && is_all_over == n_procs) {
                std::cout << "master over" << std::endl;
                break;
            }
            if (rank != MASTER_ID && is_local_over) {
                std::cout << "rank " << rank << " over" << std::endl;
                break;
            }
            */

            solver_.Pull();
        }
        if (rank != MASTER_ID) {
            fprintf(stdout, "rank=%d epoch=%zu train-loss=[%.6f]\n", rank, iter, static_cast<float>(loss) / cur_cnt);
        }
   
        file_parser.CloseFile();

        if (rank == MASTER_ID && test_file) {
            double eval_loss = evaluate_file(test_file, predict_func);
            fprintf(stdout, "validation-loss=[%lf]\n", eval_loss);
        }
    }
    return 0;
}

template<class Func>
double evaluate_file(const char* test_file, const Func& func_predict) {
    FileParser parser;
    parser.OpenFile(test_file);

    double loss = 0;
    size_t count = 0;

    auto predict_worker = [&] () {    
        std::vector<std::pair<size_t, double> > x;
        double y;
        while (1) {
            if (!parser.ReadSample(y, x))
                break;

            loss += calc_loss(y, func_predict(x));
            ++count;
        }
    };

    predict_worker();

    parser.CloseFile();
    if (count > 0) 
        loss /= count;
    return loss;
}

#endif
