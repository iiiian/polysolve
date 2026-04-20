//////////////////////////////////////////////////////////////////////////
#include <polysolve/Types.hpp>
#include <polysolve/linear/FEMSolver.hpp>

#include <polysolve/Utils.hpp>

#ifdef POLYSOLVE_WITH_AMGCL
#include <polysolve/linear/AMGCL.hpp>
#endif

#include <spdlog/sinks/stdout_color_sinks.h>

#include <catch2/catch.hpp>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <fstream>
#include <vector>
#include <ctime>
#include <chrono>
#include <cstdio>
#include "../mmio/mmio.h"
//////////////////////////////////////////////////////////////////////////

using namespace polysolve;
using namespace polysolve::linear;

void loadSymmetric(Eigen::SparseMatrix<double> &A, std::string PATH)
{
    std::ifstream fin(PATH);
    long int M, N, L;
    while (fin.peek() == '%')
    {
        fin.ignore(2048, '\n');
    }
    fin >> M >> N >> L;
    A.resize(M, N);
    A.reserve(L * 2 - M);
    std::vector<Eigen::Triplet<double>> triple;
    for (size_t i = 0; i < L; i++)
    {
        int m, n;
        double data;
        fin >> m >> n >> data;
        triple.push_back(Eigen::Triplet<double>(m - 1, n - 1, data));
        if (m != n)
        {
            triple.push_back(Eigen::Triplet<double>(n - 1, m - 1, data));
        }
    }
    fin.close();
    A.setFromTriplets(triple.begin(), triple.end());
};

bool load_matrix_market_sparse_mmio(
    const std::string &path, Eigen::SparseMatrix<double> &A)
{
    FILE *file = fopen(path.c_str(), "r");
    if (file == nullptr)
    {
        return false;
    }

    MM_typecode matcode;
    if (mm_read_banner(file, &matcode) != 0 || !mm_is_matrix(matcode) || !mm_is_coordinate(matcode)
        || !mm_is_real(matcode))
    {
        fclose(file);
        return false;
    }

    int rows = 0;
    int cols = 0;
    int nnz = 0;
    if (mm_read_mtx_crd_size(file, &rows, &cols, &nnz) != 0)
    {
        fclose(file);
        return false;
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(mm_is_symmetric(matcode) ? nnz * 2 : nnz);
    for (int index = 0; index < nnz; ++index)
    {
        int row = 0;
        int col = 0;
        double value = 0.0;
        if (fscanf(file, "%d %d %lg", &row, &col, &value) != 3)
        {
            fclose(file);
            return false;
        }

        row--;
        col--;
        triplets.emplace_back(row, col, value);
        if (mm_is_symmetric(matcode) && row != col)
        {
            triplets.emplace_back(col, row, value);
        }
    }

    fclose(file);
    A.resize(rows, cols);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return true;
}

bool load_matrix_market_array_mmio(const std::string &path, Eigen::VectorXd &b)
{
    FILE *file = fopen(path.c_str(), "r");
    if (file == nullptr)
    {
        return false;
    }

    MM_typecode matcode;
    if (mm_read_banner(file, &matcode) != 0 || !mm_is_matrix(matcode) || !mm_is_array(matcode)
        || !mm_is_real(matcode))
    {
        fclose(file);
        return false;
    }

    int rows = 0;
    int cols = 0;
    if (mm_read_mtx_array_size(file, &rows, &cols) != 0)
    {
        fclose(file);
        return false;
    }

    if (cols != 1)
    {
        fclose(file);
        return false;
    }

    b.resize(rows);
    for (int row = 0; row < rows; ++row)
    {
        if (fscanf(file, "%lg", &b[row]) != 1)
        {
            fclose(file);
            return false;
        }
    }

    fclose(file);
    return true;
}

TEST_CASE("jse", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test_logger");
    logger->set_level(spdlog::level::warn);

    json input = {};
    auto solver = Solver::create(input, *logger);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);
    const double err = (A * x - b).norm();
    INFO("solver: " + solver->name());
    REQUIRE(err < 1e-8);
}

TEST_CASE("multi-solver", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test_logger");
    logger->set_level(spdlog::level::warn);

    json input = {};
    input["solver"] = {"Hypre", "Eigen::SimplicialLDLT"};
    auto solver = Solver::create(input, *logger);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);
    const double err = (A * x - b).norm();
    INFO("solver: " + solver->name());
    REQUIRE(err < 1e-8);
}

TEST_CASE("all", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);
    json solver_info;
    Eigen::MatrixXd A_dense(A);

    auto solvers = Solver::available_solvers();
    for (const auto &s : solvers)
    {
        std::cout << s << std::endl;
    }

    for (const auto &s : solvers)
    {
        if (s == "Eigen::DGMRES")
            continue;
#ifdef WIN32
        if (s == "Eigen::ConjugateGradient" || s == "Eigen::BiCGSTAB" || s == "Eigen::GMRES" || s == "Eigen::MINRES")
            continue;
#endif
        auto solver = Solver::create(s, "");
        json params;
        params[s]["tolerance"] = 1e-10;
        solver->set_parameters(params);
        if (s == "CUDA_PCG")
        {
            params[s]["relative_tolerance"] = 0.0;
            params[s]["absolute_tolerance"] = 1e-8;
            params[s]["use_preconditioned_residual_norm"] = false;
            solver->set_parameters(params);
        }
        Eigen::VectorXd b(A.rows());
        b.setRandom();
        Eigen::VectorXd x(b.size());
        x.setZero();

        if (solver->is_dense())
        {
            solver->analyze_pattern_dense(A, A.rows());
            solver->factorize_dense(A);
        }
        else
        {
            solver->analyze_pattern(A, A.rows());
            solver->factorize(A);
        }

        solver->solve(b, x);

        REQUIRE(solver->name() == s);

        solver->get_info(solver_info);

        // std::cout<<"Solver error: "<<x<<std::endl;
        const double err = (A * x - b).norm();
        INFO("solver: " + s);
        REQUIRE(err < 1e-8);
    }
}

TEST_CASE("eigen_params", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    auto solvers = Solver::available_solvers();

    for (const auto &s : solvers)
    {
        if (s == "Eigen::ConjugateGradient" || s == "Eigen::BiCGSTAB" || s == "Eigen::GMRES" || s == "Eigen::MINRES" || s == "Eigen::LeastSquaresConjugateGradient" || s == "Eigen::DGMRES")
        {
            auto solver = Solver::create(s, "");
            json params;
            params[s]["max_iter"] = 1000;
            params[s]["tolerance"] = 1e-10;
            solver->set_parameters(params);

            Eigen::VectorXd b(A.rows());
            b.setRandom();
            Eigen::VectorXd x(b.size());
            x.setZero();

            solver->analyze_pattern(A, A.rows());
            solver->factorize(A);
            solver->solve(b, x);

            // solver->get_info(solver_info);

            // std::cout<<"Solver error: "<<x<<std::endl;
            const double err = (A * x - b).norm();
            INFO("solver: " + s);
            REQUIRE(err < 1e-8);
        }
    }
}

TEST_CASE("cuda_pcg_block_dims", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    for (int block_dim : {1, 2, 3})
    {
        auto solver = Solver::create("CUDA_PCG", "");
        json params;
        params["CUDA_PCG"]["block_dim"] = block_dim;
        params["CUDA_PCG"]["relative_tolerance"] = 0.0;
        params["CUDA_PCG"]["absolute_tolerance"] = 1e-8;
        params["CUDA_PCG"]["use_preconditioned_residual_norm"] = false;
        solver->set_parameters(params);

        Eigen::VectorXd b(A.rows());
        b.setRandom();
        Eigen::VectorXd x(b.size());
        x.setZero();

        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);

        const double err = (A * x - b).norm();
        INFO("block_dim: " + std::to_string(block_dim));
        REQUIRE(err < 1e-8);
    }
}

TEST_CASE("cuda_pcg_default_block_dim", "[solver]")
{
    auto solver = Solver::create("CUDA_PCG", "");
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    json params;
    params["CUDA_PCG"]["relative_tolerance"] = 0.0;
    params["CUDA_PCG"]["absolute_tolerance"] = 1e-8;
    params["CUDA_PCG"]["use_preconditioned_residual_norm"] = false;
    solver->set_parameters(params);

    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);

    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

TEST_CASE("cuda_pcg_pre_factor", "[solver]")
{
    auto solver = Solver::create("CUDA_PCG", "");
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    json params;
    params["CUDA_PCG"]["relative_tolerance"] = 0.0;
    params["CUDA_PCG"]["absolute_tolerance"] = 1e-8;
    params["CUDA_PCG"]["use_preconditioned_residual_norm"] = false;
    solver->set_parameters(params);
    solver->analyze_pattern(A, A.rows());

    std::default_random_engine eng{42};
    std::uniform_real_distribution<double> urd(0.1, 5);

    for (int iter = 0; iter < 3; ++iter)
    {
        std::vector<Eigen::Triplet<double>> triplet_list;
        for (int k = 0; k < A.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
            {
                if (it.row() == it.col())
                {
                    triplet_list.emplace_back(it.row(), it.col(), urd(eng) * 100);
                }
                else if (it.row() < it.col())
                {
                    double val = -urd(eng);
                    triplet_list.emplace_back(it.row(), it.col(), val);
                    triplet_list.emplace_back(it.col(), it.row(), val);
                }
            }
        }

        Eigen::SparseMatrix<double> Atmp(A.rows(), A.cols());
        Atmp.setFromTriplets(triplet_list.begin(), triplet_list.end());

        Eigen::VectorXd b(Atmp.rows());
        b.setRandom();
        Eigen::VectorXd x(b.size());
        x.setZero();

        solver->factorize(Atmp);
        solver->solve(b, x);

        const double err = (Atmp * x - b).norm();
        REQUIRE(err < 1e-8);
    }
}

TEST_CASE("cuda_pcg_some_ls_data_block3", "[solver]")
{
    constexpr auto matrix_path = "/home/ian/local_code/polysolve/some_ls_data/frame_00040_ns_0_A.mtx";
    constexpr auto rhs_path = "/home/ian/local_code/polysolve/some_ls_data/frame_00040_ns_0_b.mtx";

    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
    REQUIRE(load_matrix_market_sparse_mmio(matrix_path, A));
    REQUIRE(load_matrix_market_array_mmio(rhs_path, b));
    REQUIRE(A.rows() == b.size());

    auto solver = Solver::create("CUDA_PCG", "");
    json params;
    params["CUDA_PCG"]["block_dim"] = 3;
    params["CUDA_PCG"]["relative_tolerance"] = 0.0;
    params["CUDA_PCG"]["absolute_tolerance"] = 1e-6;
    params["CUDA_PCG"]["use_preconditioned_residual_norm"] = false;
    solver->set_parameters(params);

    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);

    const double err = (A * x - b).norm();
    INFO("err: " + std::to_string(err));
    REQUIRE(err < 1e-6);
}

TEST_CASE("pre_factor", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    auto solvers = Solver::available_solvers();

    for (const auto &s : solvers)
    {
        if (s == "Eigen::DGMRES")
            continue;
#ifdef WIN32
        if (s == "Eigen::ConjugateGradient" || s == "Eigen::BiCGSTAB" || s == "Eigen::GMRES" || s == "Eigen::MINRES")
            continue;
#endif
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        auto solver = Solver::create(s, "");
        solver->analyze_pattern(A, A.rows());

        std::default_random_engine eng{42};
        std::uniform_real_distribution<double> urd(0.1, 5);

        for (int i = 0; i < 10; ++i)
        {
            std::vector<Eigen::Triplet<double>> tripletList;

            for (int k = 0; k < A.outerSize(); ++k)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
                {
                    if (it.row() == it.col())
                    {
                        tripletList.emplace_back(it.row(), it.col(), urd(eng) * 100);
                    }
                    else if (it.row() < it.col())
                    {
                        const double val = -urd(eng);
                        tripletList.emplace_back(it.row(), it.col(), val);
                        tripletList.emplace_back(it.col(), it.row(), val);
                    }
                }
            }

            Eigen::SparseMatrix<double> Atmp(A.rows(), A.cols());
            Atmp.setFromTriplets(tripletList.begin(), tripletList.end());

            Eigen::VectorXd b(Atmp.rows());
            b.setRandom();
            Eigen::VectorXd x(b.size());
            x.setZero();

            solver->factorize(Atmp);
            solver->solve(b, x);

            // solver->get_info(solver_info);

            // std::cout<<"Solver error: "<<x<<std::endl;
            const double err = (Atmp * x - b).norm();
            INFO("solver: " + s);
            REQUIRE(err < 1e-8);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << s << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    }
}

TEST_CASE("hypre", "[solver]")
{
    std::unique_ptr<Solver> solver;

    try
    {
        solver = Solver::create("Hypre", "");
    }
    catch (const std::exception &)
    {
        return;
    }
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    // solver->set_parameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);

    // solver->get_info(solver_info);

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

TEST_CASE("hypre_initial_guess", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    // solver->set_parameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(A.rows());
    x.setZero();
    {
        json solver_info;
        std::unique_ptr<Solver> solver;
        try
        {
            solver = Solver::create("Hypre", "");
        }
        catch (const std::exception &)
        {
            return;
        }
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);
        solver->get_info(solver_info);

        REQUIRE(solver_info["num_iterations"] > 1);
    }

    {
        json solver_info;
        std::unique_ptr<Solver> solver;

        try
        {
            solver = Solver::create("Hypre", "");
        }
        catch (const std::exception &)
        {
            return;
        }
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);

        solver->get_info(solver_info);

        REQUIRE(solver_info["num_iterations"] == 1);
    }

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

TEST_CASE("amgcl_initial_guess", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    // solver->set_parameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(A.rows());
    x.setZero();
    {
        json solver_info;
        std::unique_ptr<Solver> solver;

        try
        {
            solver = Solver::create("AMGCL", "");
        }
        catch (const std::exception &)
        {
            return;
        }
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);
        solver->get_info(solver_info);

        REQUIRE(solver_info["num_iterations"] > 0);
    }

    {
        json solver_info;
        std::unique_ptr<Solver> solver;
        try
        {
            solver = Solver::create("AMGCL", "");
        }
        catch (const std::exception &)
        {
            return;
        }
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);

        solver->get_info(solver_info);

        REQUIRE(solver_info["num_iterations"] == 0);
    }

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

TEST_CASE("saddle_point_test", "[solver]")
{
#ifdef WIN32
#ifndef NDEBUG
    return;
#endif
#endif
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    bool ok = loadMarket(A, path + "/A0.mat");
    REQUIRE(ok);

    Eigen::VectorXd b;
    ok = loadMarketVector(b, path + "/b0.mat");
    REQUIRE(ok);

    auto solver = Solver::create("SaddlePointSolver", "");
    solver->analyze_pattern(A, 9934);
    solver->factorize(A);
    Eigen::VectorXd x(A.rows());
    solver->solve(b, x);

    json solver_info;
    solver->get_info(solver_info);

    REQUIRE(solver->name() == "SaddlePointSolver");

    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

#ifdef POLYSOLVE_WITH_AMGCL
TEST_CASE("amgcl_blocksolver_small_scale", "[solver]")
{
#ifndef NDEBUG
    return;
#endif
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    // solver->set_parameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(A.rows());
    Eigen::VectorXd x_b(A.rows());
    x.setZero();
    x_b.setZero();
    {
        json solver_info;

        auto solver = Solver::create("AMGCL", "");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);
        solver->get_info(solver_info);

        REQUIRE(solver_info["num_iterations"] > 0);
        const double err = (A * x - b).norm();
        REQUIRE(err < 1e-5);
    }

    {
        json solver_info;
        auto solver = Solver::create("AMGCL", "");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["block_size"] = 3;
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x_b);
        solver->get_info(solver_info);

        REQUIRE(solver_info["num_iterations"] > 0);
        const double err = (A * x_b - b).norm();
        REQUIRE(err < 1e-5);
    }
}

TEST_CASE("amgcl_blocksolver_b2", "[solver]")
{
#ifndef NDEBUG
    return;
#endif
    const std::string path = POLYFEM_DATA_DIR;
    std::string MatrixName = "gr_30_30.mtx";
    Eigen::SparseMatrix<double> A;
    loadSymmetric(A, path + "/" + MatrixName);

    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(A.rows());
    Eigen::VectorXd x_b(A.rows());
    x.setOnes();
    x_b.setOnes();
    {
        amgcl::profiler<> prof("gr_30_30_Scalar");
        json solver_info;
        auto solver = Solver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 1000;
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x);
        prof.toc("solve");
        solver->get_info(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    {
        amgcl::profiler<> prof("gr_30_30_Block");
        json solver_info;
        auto solver = Solver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 1000;
        params["AMGCL"]["block_size"] = 2;
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x_b);
        prof.toc("solve");
        solver->get_info(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    REQUIRE((A * x - b).norm() / b.norm() < 1e-7);
    REQUIRE((A * x_b - b).norm() / b.norm() < 1e-7);
}

TEST_CASE("amgcl_blocksolver_crystm03_CG", "[solver]")
{
#ifndef NDEBUG
    return;
#endif
    std::cout << "Polysolve AMGCL Solver" << std::endl;
    const std::string path = POLYFEM_DATA_DIR;
    std::string MatrixName = "crystm03.mtx";
    Eigen::SparseMatrix<double> A;
    loadSymmetric(A, path + "/" + MatrixName);
    Eigen::VectorXd b(A.rows());
    b.setOnes();
    Eigen::VectorXd x_b(A.rows());
    x_b.setZero();
    Eigen::VectorXd x(A.rows());
    x.setZero();
    {
        amgcl::profiler<> prof("crystm03_Block");
        json solver_info;
        auto solver = Solver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 1000;
        params["AMGCL"]["block_size"] = 3;
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x_b);
        prof.toc("solve");
        solver->get_info(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    {
        amgcl::profiler<> prof("crystm03_Scalar");
        json solver_info;
        auto solver = Solver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 10000;
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x);
        prof.toc("solve");
        solver->get_info(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    REQUIRE((A * x - b).norm() / b.norm() < 1e-7);
    REQUIRE((A * x_b - b).norm() / b.norm() < 1e-7);
}

TEST_CASE("amgcl_blocksolver_crystm03_Bicgstab", "[solver]")
{
#ifndef NDEBUG
    return;
#endif
    std::cout << "Polysolve AMGCL Solver" << std::endl;
    const std::string path = POLYFEM_DATA_DIR;
    std::string MatrixName = "crystm03.mtx";
    Eigen::SparseMatrix<double> A;
    loadSymmetric(A, path + "/" + MatrixName);

    Eigen::VectorXd b(A.rows());
    b.setOnes();
    Eigen::VectorXd x_b(A.rows());
    x_b.setZero();
    Eigen::VectorXd x(A.rows());
    x.setZero();
    {
        amgcl::profiler<> prof("crystm03_Block");
        json solver_info;
        auto solver = Solver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 10000;
        params["AMGCL"]["block_size"] = 3;
        params["AMGCL"]["solver_type"] = "bicgstab";
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x_b);
        prof.toc("solve");
        solver->get_info(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    {
        amgcl::profiler<> prof("crystm03_Scalar");
        json solver_info;
        auto solver = Solver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 10000;
        params["AMGCL"]["solver_type"] = "bicgstab";
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x);
        prof.toc("solve");
        solver->get_info(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    REQUIRE((A * x - b).norm() / b.norm() < 1e-7);
    REQUIRE((A * x_b - b).norm() / b.norm() < 1e-7);
}
#endif

TEST_CASE("cusolverdn", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);
    std::unique_ptr<Solver> solver;
    try
    {
        solver = Solver::create("cuSolverDN", "");
    }
    catch (const std::exception &)
    {
        return;
    }
    // solver->set_parameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

TEST_CASE("cusolverdn_dense", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;

    Eigen::MatrixXd A(4, 4);
    for (int i = 0; i < 4; i++)
    {
        A(i, i) = 1.0;
    }
    A(0, 1) = 1.0;
    A(3, 0) = 1.0;
    std::unique_ptr<Solver> solver;
    try
    {
        solver = Solver::create("cuSolverDN", "");
    }
    catch (const std::exception &)
    {
        return;
    }
    // solver->set_parameters(params);
    for (int i = 0; i < 5; ++i)
    {
        Eigen::VectorXd b(A.rows());
        b.setRandom();
        Eigen::VectorXd x(b.size());
        x.setZero();

        solver->analyze_pattern_dense(A, A.rows());
        solver->factorize_dense(A);
        solver->solve(b, x);

        // std::cout<<"Solver error: "<<x<<std::endl;
        const double err = (A * x - b).norm();
        REQUIRE(err < 1e-8);
    }
}

TEST_CASE("cusolverdn_dense_float", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;

    Eigen::MatrixXd A(4, 4);
    for (int i = 0; i < 4; i++)
    {
        A(i, i) = 1.0;
    }
    A(0, 1) = 1.0;
    A(3, 0) = 1.0;
    std::unique_ptr<Solver> solver;
    try
    {
        solver = Solver::create("cuSolverDN_float", "");
    }
    catch (const std::exception &)
    {
        return;
    }
    // solver->set_parameters(params);

    for (int i = 0; i < 5; ++i)
    {
        Eigen::VectorXd b(A.rows());
        b.setRandom();
        Eigen::VectorXd x(b.size());
        x.setZero();

        solver->analyze_pattern_dense(A, A.rows());
        solver->factorize_dense(A);
        solver->solve(b, x);

        // std::cout<<"Solver error: "<<x<<std::endl;
        const double err = (A * x - b).norm();
        REQUIRE(err < 1e-6);
    }
}

TEST_CASE("cusolverdn_5cubes", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;

    std::unique_ptr<Solver> solver;
    try
    {
        solver = Solver::create("cuSolverDN", "");
    }
    catch (const std::exception &)
    {
        return;
    }

    // std::ofstream factorize_times_file(path+"/factorize_times_5cubes.txt");
    // std::ofstream solve_times_file(path+"/solve_times_5cubes.txt");

    for (int i = 0; i <= 1091; i++)
    {
        Eigen::MatrixXd A(120, 120);
        std::string hessian_path = path + "/matrixdata-5cubes/hessian" + std::to_string(i) + ".txt";
        std::ifstream hessian_file(hessian_path);
        for (int m = 0; m < 120; m++)
        {
            for (int n = 0; n < 120; n++)
            {
                hessian_file >> A(m, n);
            }
        }

        Eigen::VectorXd b(A.rows());
        std::string gradient_path = path + "/matrixdata-5cubes/gradient" + std::to_string(i) + ".txt";
        std::ifstream gradient_file(gradient_path);
        for (int m = 0; m < 120; m++)
        {
            gradient_file >> b(m);
        }

        Eigen::VectorXd x(b.size());
        x.setZero();

        solver->analyze_pattern_dense(A, A.rows());

        // std::chrono::steady_clock::time_point beginf = std::chrono::steady_clock::now();
        solver->factorize_dense(A);
        // std::chrono::steady_clock::time_point endf = std::chrono::steady_clock::now();
        // std::cout << "time to factorize: " << std::chrono::duration_cast<std::chrono::nanoseconds>(endf-beginf).count() << std::endl;
        // factorize_times_file << std::chrono::duration_cast<std::chrono::nanoseconds>(endf-beginf).count() << " ";

        // std::chrono::steady_clock::time_point begins = std::chrono::steady_clock::now();
        solver->solve(b, x);
        // std::chrono::steady_clock::time_point ends = std::chrono::steady_clock::now();
        // std::cout << "time to solve: " << std::chrono::duration_cast<std::chrono::nanoseconds>(ends-begins).count() << std::endl;
        // solve_times_file << std::chrono::duration_cast<std::chrono::nanoseconds>(ends-begins).count() << " ";

        // std::cout << "Ax norm: " << (A*x).norm() << std::endl;
        // std::cout << "b norm: " << b.norm() << std::endl;

        const double err = (A * x - b).norm();
        REQUIRE(err < 1e-8);
    }
}
