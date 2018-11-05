#include <getopt.h>
#include <unistd.h>
#include <iostream>
#include <locale>
#include "ftrl_train.h"

using namespace std;
const int MAX_BUF_LEN = 2048;

void print_usage() {
	printf("Usage: ./ftrl_train -f input_file -m model_file [options]\n"
		"options:\n"
		"-f input_file : set train file. You can read sample from stdin by set '-f stdin'\n"
		"-t test_file : set evaluation file\n"
		// "--cache/-c : cache feature count and sample count of input file, default true\n"
		"--epoch iteration : set number of iteration, default 1\n"
		"--alpha alpha : set alpha param, default 0.15\n"
		"--beta beta : set beta param, default 1\n"
		"--l1 l1 : set l1 param, default 1\n"
		"--l2 l2 : set l2 param, default 1\n"
		"--help : print this help\n"
	);
}


bool train(const char* input_file, const char* test_file, const char* model_file,
		double alpha, double beta, double l1, double l2,
		size_t epoch, uint32_t mini_batch) {
        int rank, n_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

        size_t local_feat_num = 0, feat_num = 0;
        size_t local_line_cnt = 0, line_cnt = 0;

	FtrlTrainer trainer;

        char local_train_file[MAX_BUF_LEN];
        snprintf(local_train_file, MAX_BUF_LEN, "%s_%05d", input_file, rank);
        if (rank != MASTER_ID) {
            local_feat_num = read_problem_info(local_train_file, local_line_cnt, rank);
            if (local_feat_num == 0) {
                fprintf(stdout, "read_problem_info failed\n");
                return false;
            }
        }
        MPI_Reduce(&local_feat_num, &feat_num, n_procs, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_line_cnt, &line_cnt, n_procs, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Bcast(&feat_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&line_cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        fprintf(stdout, "rank %d: feat_num=%lu, line_cnt=%lu\n", rank,
                feat_num, line_cnt);
	trainer.Initialize(epoch, feat_num, alpha, beta, l1, l2);
	trainer.Train(alpha, beta, l1, l2, feat_num, line_cnt,
                model_file, local_train_file, test_file, mini_batch);

	return true;
}

int main(int argc, char* argv[]) {
        MPI_Init(&argc, &argv);
	int opt;
	int opt_idx = 0;

	static struct option long_options[] = {
		{"epoch", required_argument, NULL, 'i'},
		{"alpha", required_argument, NULL, 'a'},
		{"beta", required_argument, NULL, 'b'},
		{"l1", required_argument, NULL, 'l'},
		{"l2", required_argument, NULL, 'e'},
		{"feat-num", required_argument, NULL, 'k'},
                {"mini_batch", no_argument, NULL, 'M'},
		{"help", no_argument, NULL, 'h'},
		{0, 0, 0, 0}
	};

	std::string input_file;
	std::string test_file;
	std::string model_file;
	std::string start_from_model;

	double alpha = DEFAULT_ALPHA;
	double beta = DEFAULT_BETA;
	double l1 = DEFAULT_L1;
	double l2 = DEFAULT_L2;

	size_t epoch = 1;

        uint32_t mini_batch = 0;

	while ((opt = getopt_long(argc, argv, "f:t:m:M:ch", long_options, &opt_idx)) != -1) {
		switch (opt) {
		case 'f':
			input_file = optarg;
			break;
		case 't':
			test_file = optarg;
			break;
		case 'm':
			model_file = optarg;
			break;
		case 'i':
			epoch = (size_t)atoi(optarg);
			break;
		case 'a':
			alpha = atof(optarg);
			break;
		case 'b':
			beta = atof(optarg);
			break;
		case 'l':
			l1 = atof(optarg);
			break;
		case 'e':
			l2 = atof(optarg);
			break;
                case 'M':
                        mini_batch = atoi(optarg);
                        break;
		case 'h':
		default:
			print_usage();
			exit(0);
		}
	}

	if (input_file.size() == 0 || model_file.size() == 0) {
		print_usage();
		exit(1);
	}

	const char* ptest_file = NULL;
	if (test_file.size() > 0) ptest_file = test_file.c_str();

	train(input_file.c_str(), ptest_file, model_file.c_str(),
		alpha, beta, l1, l2, epoch, mini_batch);

        MPI_Finalize();
	return 0;
}
/* vim: set ts=4 sw=4 tw=0 noet :*/

