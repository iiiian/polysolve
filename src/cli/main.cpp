#include <polysolve/Types.hpp>
#include <polysolve/Utils.hpp>
#include <polysolve/linear/Solver.hpp>

#include <argparse/argparse.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <Eigen/Sparse>

#include <chrono>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <regex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

extern "C"
{
#include "mmio.h"
}

using polysolve::json;
using polysolve::StiffnessMatrix;

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point &start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

json read_json_file(const std::string &path)
{
    std::ifstream in(path);
    if (!in.is_open())
    {
        throw std::runtime_error("Unable to open json file: " + path);
    }
    json j;
    in >> j;
    return j;
}

StiffnessMatrix load_sparse_matrix_market(const std::string &path)
{
    FILE *f = fopen(path.c_str(), "r");
    if (!f)
    {
        throw std::runtime_error("Unable to open A file: " + path);
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0)
    {
        fclose(f);
        throw std::runtime_error("Unable to read MatrixMarket banner for A: " + path);
    }

    if (!(mm_is_matrix(matcode) && mm_is_coordinate(matcode) && mm_is_real(matcode)))
    {
        fclose(f);
        throw std::runtime_error("A must be MatrixMarket 'matrix coordinate real': " + path);
    }

    int M = 0, N = 0, nz = 0;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0)
    {
        fclose(f);
        throw std::runtime_error("Unable to read MatrixMarket coordinate size for A: " + path);
    }
    if (M != N)
    {
        fclose(f);
        throw std::runtime_error("A must be square; got " + std::to_string(M) + "x" + std::to_string(N));
    }

    bool is_symmetric = mm_is_symmetric(matcode) != 0;

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(is_symmetric ? 2 * nz : nz);

    for (int k = 0; k < nz; ++k)
    {
        int i = 0, j = 0;
        double v = 0.0;
        double imag = 0.0;
        int rc = mm_read_mtx_crd_entry(f, &i, &j, &v, &imag, matcode);
        if (rc != 0)
        {
            fclose(f);
            throw std::runtime_error("Failed reading A entry " + std::to_string(k) + " from: " + path);
        }
        i -= 1;
        j -= 1;
        triplets.emplace_back(i, j, v);
        if (is_symmetric && i != j)
        {
            triplets.emplace_back(j, i, v);
        }
    }

    fclose(f);

    StiffnessMatrix A(M, N);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

Eigen::VectorXd load_dense_vector_market(const std::string &path)
{
    FILE *f = fopen(path.c_str(), "r");
    if (!f)
    {
        throw std::runtime_error("Unable to open b file: " + path);
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0)
    {
        fclose(f);
        throw std::runtime_error("Unable to read MatrixMarket banner for b: " + path);
    }

    if (!(mm_is_matrix(matcode) && mm_is_array(matcode) && mm_is_real(matcode)))
    {
        fclose(f);
        throw std::runtime_error("b must be MatrixMarket 'matrix array real': " + path);
    }

    int M = 0, N = 0;
    if (mm_read_mtx_array_size(f, &M, &N) != 0)
    {
        fclose(f);
        throw std::runtime_error("Unable to read MatrixMarket array size for b: " + path);
    }
    if (N != 1)
    {
        fclose(f);
        throw std::runtime_error("b must be an n x 1 array; got " + std::to_string(M) + "x" + std::to_string(N));
    }

    Eigen::VectorXd b(M);
    for (int i = 0; i < M; ++i)
    {
        double v = 0.0;
        if (fscanf(f, "%lg", &v) != 1)
        {
            fclose(f);
            throw std::runtime_error("Failed reading b entry " + std::to_string(i) + " from: " + path);
        }
        b(i) = v;
    }

    fclose(f);
    return b;
}

// redirect stdout to a unix pipe object to capture all output
std::string capture_stdout(const std::function<void()> &fn)
{

    // flush all C/C++ output buffer
    fflush(nullptr);
    std::cout.flush();

    int pipefd[2];
    if (pipe(pipefd) != 0)
    {
        throw std::runtime_error("pipe() failed");
    }
    int pipe_read = pipefd[0];
    int pipe_write = pipefd[1];

    // saved stdout file desc then rebind it to our pipe
    int saved = dup(STDOUT_FILENO);
    if (saved < 0)
    {
        close(pipe_read);
        close(pipe_write);
        throw std::runtime_error("dup() failed");
    }
    if (dup2(pipe_write, STDOUT_FILENO) < 0)
    {
        close(pipe_read);
        close(pipe_write);
        close(saved);
        throw std::runtime_error("dup2() failed");
    }

    // we no longer need this file desc as stdout now points to this pipe.
    close(pipe_write);

    // launch another thread to read pipe into captured.
    std::string captured;
    std::thread reader([&]() {
        constexpr size_t buf_size = 4096;
        char buf[buf_size];
        ssize_t n = read(pipe_read, buf, buf_size);
        while (n > 0)
        {
            captured.append(buf, buf + n);
            n = read(pipe_read, buf, buf_size);
        }
        close(pipe_read);
    });

    fn();

    // flush all C/C++ output buffer
    fflush(nullptr);
    std::cout.flush();

    // map stdout file desc back to true stdout
    dup2(saved, STDOUT_FILENO);
    close(saved);

    reader.join();
    return captured;
}

struct IterationStats
{
    std::vector<int> iters;
    std::vector<double> residuals;
    std::vector<double> times_s;
};

struct HypreTimingStats
{
    IterationStats pcg_iter;
    IterationStats amg_cycles;
};

HypreTimingStats parse_hypre_timing_stats(const std::string &text)
{
    HypreTimingStats out;

    std::regex pcg_re(
        R"((?:^|\n)HYPRE_PCG_ITER\s+(\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))");
    for (std::sregex_iterator it(text.begin(), text.end(), pcg_re), end; it != end; ++it)
    {
        out.pcg_iter.iters.push_back(std::stoi((*it)[1].str()));
        out.pcg_iter.residuals.push_back(std::stod((*it)[2].str()));
        out.pcg_iter.times_s.push_back(std::stod((*it)[4].str()));
    }

    std::regex amg_re(
        R"((?:^|\n)HYPRE_AMG_CYCLE\s+(\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))");
    for (std::sregex_iterator it(text.begin(), text.end(), amg_re), end; it != end; ++it)
    {
        out.amg_cycles.iters.push_back(std::stoi((*it)[1].str()));
        out.amg_cycles.residuals.push_back(std::stod((*it)[2].str()));
        out.amg_cycles.times_s.push_back(std::stod((*it)[4].str()));
    }

    if (out.pcg_iter.iters.empty() && out.amg_cycles.iters.empty())
    {
        throw std::runtime_error("No hypre statistic logs");
    }

    return out;
}

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("polysolve");
    program.add_argument("--json").required();
    program.add_argument("--A").required();
    program.add_argument("--b").required();
    program.add_argument("--out").required();
    program.parse_args(argc, argv);

    auto logger = spdlog::stdout_color_mt("polysolve_cli");
    logger->set_level(spdlog::level::info);

    std::string json_path = program.get<std::string>("--json");
    std::string A_path = program.get<std::string>("--A");
    std::string b_path = program.get<std::string>("--b");
    std::string out_path = program.get<std::string>("--out");

    json config = read_json_file(json_path);

    StiffnessMatrix A = load_sparse_matrix_market(A_path);
    Eigen::VectorXd b = load_dense_vector_market(b_path);
    if (A.rows() != b.size())
    {
        throw std::runtime_error("Size mismatch: A is " + std::to_string(A.rows()) + "x" + std::to_string(A.cols())
                                 + " but b is " + std::to_string(b.size()) + "x1");
    }

    auto solver = polysolve::linear::Solver::create(config, *logger);

    if (solver->name() != "CUDA_PCG" && solver->name() != "Hypre" && solver->name() != "Experimental"
        && solver->name() != "Eigen::PardisoLDLT")
    {
        throw std::runtime_error(
            "CLI only supports solvers: CUDA_PCG, Hypre, Experimental, Eigen::PardisoLDLT (selected: " + solver->name() + ")");
    }

    Eigen::VectorXd x(b.size());
    x.setZero();

    const Clock::time_point analyze_start = Clock::now();
    solver->analyze_pattern(A, A.rows());
    const double analyze_s = seconds_since(analyze_start);

    const Clock::time_point fact_start = Clock::now();
    solver->factorize(A);
    const double factorize_s = seconds_since(fact_start);

    HypreTimingStats hypre_timing;
    double solve_s = 0.0;
    // Hijiack and parse stdout to read custom hypre log.
    if (solver->name() == "Hypre")
    {
        solver->set_parameters(config);

        const Clock::time_point solve_start = Clock::now();
        const auto captured = capture_stdout([&]() { solver->solve(b, x); });
        solve_s = seconds_since(solve_start);

        hypre_timing = parse_hypre_timing_stats(captured);
    }
    // For other solver we can get statistics from solver info.
    else
    {
        const Clock::time_point solve_start = Clock::now();
        solver->solve(b, x);
        solve_s = seconds_since(solve_start);
    }

    const double residual_norm = (A * x - b).norm();

    json solver_info;
    solver->get_info(solver_info);

    json all_json_stats;
    all_json_stats["input"] = {
        {"json", json_path},
        {"A", A_path},
        {"b", b_path},
        {"out", out_path},
    };
    all_json_stats["solver"] = {
        {"name", solver->name()},
        {"info", solver_info},
    };
    all_json_stats["timing_s"] = {
        {"analyze", analyze_s},
        {"factorize", factorize_s},
        {"solve", solve_s},
    };
    all_json_stats["residual_norm"] = residual_norm;

    if (solver->name() == "Hypre")
    {
        all_json_stats["solver"]["iteration_stats"] = {
            {"iteration", hypre_timing.pcg_iter.iters},
            {"residual", hypre_timing.pcg_iter.residuals},
            {"time_s", hypre_timing.pcg_iter.times_s},
        };
        all_json_stats["solver"]["amg_cycle_stats"] = {
            {"cycle", hypre_timing.amg_cycles.iters},
            {"residual", hypre_timing.amg_cycles.residuals},
            {"time_s", hypre_timing.amg_cycles.times_s},
        };
    }

    // write json stat to file
    std::ofstream json_out(out_path);
    if (!json_out.is_open())
    {
        throw std::runtime_error("Failed to open output file: " + out_path);
    }
    json_out << all_json_stats.dump() << "\n";

    // human readable logging
    logger->info("solver: {}", solver->name());
    logger->info("timing [s]: analyze={} factorize={} solve={}", analyze_s, factorize_s, solve_s);
    logger->info("residual ||Ax-b|| = {}", residual_norm);
    if (solver_info.contains("solver_iter"))
    {
        logger->info("solver_iter: {}", solver_info["solver_iter"].dump());
    }
    if (solver_info.contains("solver_error"))
    {
        logger->info("solver_error: {}", solver_info["solver_error"].dump());
    }
    if (solver_info.contains("solver_status"))
    {
        logger->info("solver_status: {}", solver_info["solver_status"].dump());
    }
    if (solver_info.contains("iteration_residual_norm") && solver_info.contains("iteration_time_s"))
    {
        if (!solver_info.contains("solver_iter"))
        {
            throw std::runtime_error("solver_info missing required key: solver_iter");
        }
        auto &res = solver_info["iteration_residual_norm"];
        auto &ts = solver_info["iteration_time_s"];
        const size_t expected = static_cast<size_t>(solver_info["solver_iter"].get<int>()) + 1;
        if (!res.is_array() || !ts.is_array() || res.size() != expected || ts.size() != expected)
        {
            throw std::runtime_error("iteration stats must be arrays sized solver_iter+1");
        }
        for (size_t i = 0; i < res.size(); ++i)
        {
            logger->info("iter {} residual {} dt_s {}", static_cast<int>(i), res[i].get<double>(), ts[i].get<double>());
        }
    }

    return 0;
}
