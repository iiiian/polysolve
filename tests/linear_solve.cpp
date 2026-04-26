#include <polysolve/Types.hpp>
#include <polysolve/linear/Solver.hpp>

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <unsupported/Eigen/SparseExtra>

#include <cstdio>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>

using namespace polysolve;
using namespace polysolve::linear;

namespace
{
    class ScopedOutputSilencer
    {
    public:
        ScopedOutputSilencer()
        {
            flush_output();

            null_fd_ = open("/dev/null", O_WRONLY);
            if (null_fd_ == -1)
            {
                throw std::runtime_error("failed to open /dev/null");
            }

            stdout_fd_ = dup(STDOUT_FILENO);
            if (stdout_fd_ == -1)
            {
                close(null_fd_);
                throw std::runtime_error("failed to duplicate stdout");
            }

            stderr_fd_ = dup(STDERR_FILENO);
            if (stderr_fd_ == -1)
            {
                close(stdout_fd_);
                close(null_fd_);
                throw std::runtime_error("failed to duplicate stderr");
            }

            if (dup2(null_fd_, STDOUT_FILENO) == -1 || dup2(null_fd_, STDERR_FILENO) == -1)
            {
                restore();
                throw std::runtime_error("failed to redirect output to /dev/null");
            }
        }

        ~ScopedOutputSilencer()
        {
            restore();
        }

        ScopedOutputSilencer(const ScopedOutputSilencer &) = delete;
        ScopedOutputSilencer &operator=(const ScopedOutputSilencer &) = delete;

    private:
        void restore() noexcept
        {
            flush_output();

            if (stdout_fd_ != -1)
            {
                (void)dup2(stdout_fd_, STDOUT_FILENO);
                close(stdout_fd_);
                stdout_fd_ = -1;
            }
            if (stderr_fd_ != -1)
            {
                (void)dup2(stderr_fd_, STDERR_FILENO);
                close(stderr_fd_);
                stderr_fd_ = -1;
            }
            if (null_fd_ != -1)
            {
                close(null_fd_);
                null_fd_ = -1;
            }
        }

        static void flush_output() noexcept
        {
            std::fflush(stdout);
            std::fflush(stderr);
        }

        int null_fd_ = -1;
        int stdout_fd_ = -1;
        int stderr_fd_ = -1;
    };

    Eigen::VectorXd make_random_rhs(const Eigen::Index size, const std::optional<std::uint32_t> seed)
    {
        std::mt19937 generator;
        if (seed.has_value())
        {
            generator.seed(*seed);
        }
        else
        {
            std::random_device random_device;
            generator.seed(random_device());
        }

        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        Eigen::VectorXd rhs(size);
        for (Eigen::Index i = 0; i < size; ++i)
        {
            rhs[i] = distribution(generator);
        }
        return rhs;
    }
} // namespace

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("linear_solve");
    program.add_description("Run a PolySolve linear solve from Matrix Market inputs.");
    program.add_argument("-A")
        .required()
        .metavar("matrix.mtx")
        .help("Input Matrix Market sparse matrix.");
    auto &rhs_group = program.add_mutually_exclusive_group();
    rhs_group.add_argument("-b")
        .metavar("rhs.mtx")
        .help("Optional Matrix Market RHS vector. Defaults to a zero vector unless --rand is used.");
    rhs_group.add_argument("--rand")
        .metavar("seed")
        .scan<'i', long long>()
        .nargs(argparse::nargs_pattern::optional)
        .help("Generate a random RHS instead of loading -b. Optionally provide a seed.");
    program.add_argument("-j")
        .metavar("spec.json")
        .help("Optional solver JSON config. Defaults to {\"solver\":\"Eigen::SimplicialLDLT\"}.");
    program.add_argument("-w")
        .default_value(1)
        .scan<'i', int>()
        .metavar("warmup")
        .help("Number of warmup iterations. Default: 1.");
    program.add_argument("-r")
        .default_value(1)
        .scan<'i', int>()
        .metavar("repeat")
        .help("Number of iterations after warmup. Default: 1.");

    program.parse_args(argc, argv);

    const std::string matrix_path = program.get<std::string>("-A");
    const int warmup = program.get<int>("-w");
    const int repeat = program.get<int>("-r");
    const bool use_random_rhs = program.is_used("--rand");
    const auto rand_seed_argument = program.present<long long>("--rand");

    if (warmup < 0)
    {
        throw std::runtime_error("warmup count must be non-negative");
    }
    if (repeat < 1)
    {
        throw std::runtime_error("repeat count must be at least 1");
    }

    std::optional<std::uint32_t> random_seed;
    if (rand_seed_argument.has_value())
    {
        if (*rand_seed_argument < 0)
        {
            throw std::runtime_error("random seed must be non-negative");
        }
        if (static_cast<unsigned long long>(*rand_seed_argument) > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error("random seed exceeds uint32_t range");
        }
        random_seed = static_cast<std::uint32_t>(*rand_seed_argument);
    }

    json solver_config = json::object({{"solver", "Eigen::SimplicialLDLT"}});
    if (program.present("-j"))
    {
        const std::string json_path = program.get<std::string>("-j");
        std::ifstream in(json_path);
        if (!in.is_open())
        {
            throw std::runtime_error("failed to open json config: " + json_path);
        }
        in >> solver_config;
    }

    auto logger = spdlog::stderr_color_mt("linear_solve");
    logger->set_level(spdlog::level::off);

    // Validate solver configuration before loading potentially large inputs.
    {
        auto solver = Solver::create(solver_config, *logger);
        (void)solver;
    }

    Eigen::SparseMatrix<double> A;
    if (!loadMarket(A, matrix_path))
    {
        throw std::runtime_error("failed to load matrix market matrix: " + matrix_path);
    }
    if (A.rows() != A.cols())
    {
        throw std::runtime_error("matrix must be square");
    }
    A.makeCompressed();

    Eigen::VectorXd b(A.rows());
    if (program.present("-b"))
    {
        const std::string rhs_path = program.get<std::string>("-b");
        if (!Eigen::loadMarketVector(b, rhs_path))
        {
            throw std::runtime_error("failed to load matrix market rhs: " + rhs_path);
        }
        if (b.size() != A.rows())
        {
            throw std::runtime_error("rhs dimension mismatch");
        }
    }
    else
    {
        if (use_random_rhs)
        {
            b = make_random_rhs(A.rows(), random_seed);
        }
        else
        {
            b.setZero();
        }
    }

    std::optional<Eigen::MatrixXd> dense_A;

    const int iterations = warmup + repeat;
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        const std::unique_ptr<ScopedOutputSilencer> silencer =
            iteration < warmup ? std::make_unique<ScopedOutputSilencer>() : nullptr;

        auto solver = Solver::create(solver_config, *logger);
        Eigen::VectorXd x(A.cols());
        x.setZero();

        if (solver->is_dense())
        {
            if (!dense_A.has_value())
            {
                dense_A.emplace(A);
            }
            solver->analyze_pattern_dense(*dense_A, A.rows());
        }
        else
        {
            solver->analyze_pattern(A, A.rows());
        }

        if (solver->is_dense())
        {
            solver->factorize_dense(*dense_A);
        }
        else
        {
            solver->factorize(A);
        }

        solver->solve(b, x);
    }

    return 0;
}
