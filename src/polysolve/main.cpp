
#include <fstream>
#include <iostream>
#include <string>
#include <ctime>
#include <format>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <stdlib.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "linear/Solver.hpp"
#include "Utils.hpp"

#include <mpi.h>
#include <mkl.h>

#include <tbb/global_control.h>

void usage(const std::string &executable)
{
    std::cout << "Usage: " << executable << " solver_type A_file b_file param_file [do_openmp_test]" << std::endl;
}


int main(int argc, char **argv)
{

    if (argc != 5 && argc != 6)
    {
        const std::string executable(argv[0]);
        usage(executable);
        return 1;
    }

    int done_already;

    MPI_Initialized(&done_already);
    int myid, num_procs;
    if (!done_already)
    {
        // Initialize MPI 
        int argc = 1;
        char name[] = "";
        char *argv[] = {name};
        char **argvv = &argv[0];
        MPI_Init(&argc, &argvv);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    }

    // Create logger
    std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt(fmt::format("solver-child-{}", myid));
    if (myid == 0)
    {
        logger->set_level(spdlog::level::trace);
    } 
    else
    {
        logger->set_level(spdlog::level::off);
    }

    const char *num_threads_c_str = std::getenv("OMP_NUM_THREADS");
    const std::string num_threads_str(
        num_threads_c_str ? num_threads_c_str : "1"
    );
    const int num_threads = std::stoi(num_threads_str);
    Eigen::setNbThreads(num_threads);

    const std::string solver_str(argv[1]);
    const std::string A_file(argv[2]);
    const std::string b_file(argv[3]);
    const std::string param_file(argv[4]);

    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
    int rows, cols, nnzs;

    if (myid == 0)
    {
        logger->info("Starting run...");

        logger->info("Solver: {} w/ params set by {}", solver_str, param_file);
        logger->info("Matrix: {}, rhs: {}", A_file, b_file);
        logger->trace("Num threads: {}", Eigen::nbThreads());

        const bool do_openmp_test = (argc == 6);

        logger->info("Loading problem...");

        double load_problem_time;
        {
            POLYSOLVE_SCOPED_STOPWATCH("load problem time", load_problem_time, *logger);

            std::ifstream A_file_stream(A_file);
            A_file_stream >> rows >> cols >> nnzs;
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(nnzs);
            A.resize(rows, cols);

            int i, j;
            double v;
            while (A_file_stream >> i >> j >> v)
            {
                triplets.push_back(Eigen::Triplet<double>(i, j, v));
            }

            A.setFromTriplets(triplets.begin(), triplets.end());
            A_file_stream.close();

            b.resize(rows);
            std::ifstream b_file_stream(b_file);

            int i_counter = 0;
            while (b_file_stream >> v)
            {
                b(i_counter++) = v;
            }
            b_file_stream.close();
        }

        logger->trace("Problem size: {}", b.size());

        if (do_openmp_test)
        {

            Eigen::SparseMatrix<double, Eigen::RowMajor> Atest = A;
            double mult_test_time, loop_test_time;
            {
                POLYSOLVE_SCOPED_STOPWATCH("multiplication test time", mult_test_time, *logger);
                double test = 0;

                for (int index = 0; index < 100; ++index)
                {
                    test += (Atest * b).maxCoeff();
                    Atest.coeffRef(0, 0) += 1;
                }
                logger->trace("Value: {}", test);
            }

            {
                POLYSOLVE_SCOPED_STOPWATCH("loop test time", loop_test_time, *logger);
                #pragma omp parallel for
                for (int index = 0; index < 1000000000; ++index)
                {
                    if (index % 900000000 == 0)
                    {
                        logger->trace("Index: {}", index);
                    }
                }
            }
        }
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnzs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (myid != 0)
    {
        A.resize(rows, cols);
        A.reserve(nnzs);

        MPI_Bcast(A.valuePtr(), nnzs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(A.innerIndexPtr(), nnzs, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(A.outerIndexPtr(), rows, MPI_INT, 0, MPI_COMM_WORLD);

        b.resize(rows);
        MPI_Bcast(b.data(), rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    logger->info("Creating solver...");
    // Solve
    auto solver = polysolve::linear::Solver::create(solver_str, "");
    solver->logger = logger.get();

    std::ifstream param_file_stream(param_file);
    polysolve::json params = polysolve::json::parse(param_file_stream);
    solver->set_parameters(params);

    Eigen::VectorXd x(b.size());
    x.setZero();

    logger->info("Starting solve...");
    double solve_time;
    int start_analyze_pattern, start_factorize, start_solve;
    if (myid == 0)
    {
        POLYSOLVE_SCOPED_STOPWATCH("total solve time", solve_time, *logger);
        MPI_Bcast(&start_analyze_pattern, 1, MPI_INT, 0, MPI_COMM_WORLD);
        solver->analyze_pattern(A, A.rows());
        MPI_Bcast(&start_factorize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        solver->factorize(A);
        MPI_Bcast(&start_solve, 1, MPI_INT, 0, MPI_COMM_WORLD);
        solver->solve(b, x);
    }
    else 
    {
        POLYSOLVE_SCOPED_STOPWATCH("total solve time", solve_time, *logger);
        MPI_Bcast(&start_analyze_pattern, 1, MPI_INT, 0, MPI_COMM_WORLD);
        solver->analyze_pattern(A, A.rows());
        MPI_Bcast(&start_factorize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        solver->factorize(A);
        MPI_Bcast(&start_solve, 1, MPI_INT, 0, MPI_COMM_WORLD);
        solver->solve(b, x);
    }

    if (myid == 0)
    {
        std::ofstream file("x.mat");
        file << x;
        file.close();
    }

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized)
        MPI_Finalize();
    
    return 0;
}