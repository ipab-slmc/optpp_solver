//
// Copyright (c) 2019, University of Edinburgh
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of  nor the names of its contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include <optpp_catkin/OptCG.h>
#include <optpp_catkin/OptFDNewton.h>
#include <optpp_catkin/OptGSS.h>
#include <optpp_catkin/OptLBFGS.h>
#include <optpp_catkin/OptQNewton.h>

#include <optpp_solver/optpp_traj.h>

REGISTER_MOTIONSOLVER_TYPE("OptppTrajLBFGS", exotica::OptppTrajLBFGS)
REGISTER_MOTIONSOLVER_TYPE("OptppTrajCG", exotica::OptppTrajCG)
REGISTER_MOTIONSOLVER_TYPE("OptppTrajQNewton", exotica::OptppTrajQNewton)
REGISTER_MOTIONSOLVER_TYPE("OptppTrajFDNewton", exotica::OptppTrajFDNewton)
REGISTER_MOTIONSOLVER_TYPE("OptppTrajGSS", exotica::OptppTrajGSS)

namespace exotica
{
template <class ProblemType, class InitializerType>
void OptppTimeIndexedSolver<ProblemType, InitializerType>::SpecifyProblem(PlanningProblemPtr pointer)
{
    if (pointer->type() != "exotica::UnconstrainedTimeIndexedProblem")
    {
        ThrowNamed("OptppTimeIndexedSolver can't solve problem of type '" << pointer->type() << "'!");
    }
    MotionSolver::SpecifyProblem(pointer);
    prob_ = std::static_pointer_cast<ProblemType>(pointer);
}

template <>
void OptppTrajLBFGS::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) ThrowNamed("Solver has not been initialized!");
    prob_->PreUpdate();
    prob_->ResetCostEvolution(GetNumberOfMaxIterations() + 1);

    solution.resize(prob_->GetT(), prob_->N);
    solution.row(0) = prob_->GetInitialTrajectory()[0];
    int iter, feval, geval, ret;
    double f_sol;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        std::shared_ptr<OPTPP::OptLBFGS> solver = nullptr;
        if (parameters_.UseFiniteDifferences)
        {
            auto nlf_local = UnconstrainedTimeIndexedProblemWrapper(prob_, true).getFDNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptLBFGS(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        else
        {
            // L-BFGS has bad handling of iterations, this is a work-around
            auto nlf_local = UnconstrainedTimeIndexedProblemWrapper(prob_, true).getNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptLBFGS(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        solver->setGradTol(parameters_.GradientTolerance);
        solver->setMaxBacktrackIter(parameters_.MaxBacktrackIterations);
        solver->setLineSearchTol(parameters_.LineSearchTolerance);
        solver->setStepTol(parameters_.StepTolerance);
        solver->setMaxIter(GetNumberOfMaxIterations());
        solver->setFcnTol(parameters_.FunctionTolerance);
        solver->setMinStep(parameters_.MinStep);
        solver->setOutputFile("/tmp/OPTPP_DEFAULT.out", 0);
        // nlf->setIsExpensive(1);  // 1 uses Simple Linesearch, 0 uses Moore-Thuente

        solver->optimize();
        ColumnVector sol = nlf->getXc();
        for (int t = 1; t < prob_->GetT(); t++)
            for (int i = 0; i < prob_->N; i++)
                solution(t, i) = sol((t - 1) * prob_->N + i + 1);
        iter = solver->getIter();
        feval = nlf->getFevals();
        geval = nlf->getGevals();
        f_sol = nlf->getF();
        ret = solver->getReturnCode();
        solver->cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        ThrowPretty("OPT++ exception:" << BaseException::what());
    }

    planning_time_ = timer.GetDuration();

    if (debug_)
    {
        HIGHLIGHT_NAMED(object_name_ + " OptppTrajLBFGS", "Time: " << planning_time_ << "s, Status: " << ret << ", Iterations: " << iter << ", Feval: " << feval << ", Geval: " << geval << ", Final cost: " << f_sol);
    }
}

template <>
void OptppTrajCG::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) ThrowNamed("Solver has not been initialized!");
    prob_->PreUpdate();
    prob_->ResetCostEvolution(GetNumberOfMaxIterations() + 1);

    solution.resize(prob_->GetT(), prob_->N);
    solution.row(0) = prob_->GetInitialTrajectory()[0];
    int iter, feval, geval, ret;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        std::shared_ptr<OPTPP::OptCG> solver(new OPTPP::OptCG());
        if (parameters_.UseFiniteDifferences)
        {
            auto nlf_local = UnconstrainedTimeIndexedProblemWrapper(prob_).getFDNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptCG(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        else
        {
            auto nlf_local = UnconstrainedTimeIndexedProblemWrapper(prob_).getNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptCG(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        solver->setGradTol(parameters_.GradientTolerance);
        solver->setMaxBacktrackIter(parameters_.MaxBacktrackIterations);
        solver->setLineSearchTol(parameters_.LineSearchTolerance);
        solver->setStepTol(parameters_.StepTolerance);
        solver->setMaxIter(GetNumberOfMaxIterations());
        solver->setFcnTol(parameters_.FunctionTolerance);
        solver->setMinStep(parameters_.MinStep);
        solver->setOutputFile("/tmp/OPTPP_DEFAULT.out", 0);
        solver->optimize();
        ColumnVector sol = nlf->getXc();
        for (int t = 1; t < prob_->GetT(); t++)
            for (int i = 0; i < prob_->N; i++)
                solution(t, i) = sol((t - 1) * prob_->N + i + 1);
        iter = solver->getIter();
        feval = nlf->getFevals();
        geval = nlf->getGevals();
        ret = solver->getReturnCode();
        solver->cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        ThrowPretty("OPT++ exception:" << BaseException::what());
    }

    planning_time_ = timer.GetDuration();

    if (debug_)
    {
        HIGHLIGHT_NAMED(object_name_ + " OptppTrajCG", "Time: " << planning_time_ << ", Status: " << ret << ", Iterations: " << iter << ", Feval: " << feval << ", Geval: " << geval);
    }
}

template <>
void OptppTrajQNewton::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) ThrowNamed("Solver has not been initialized!");
    prob_->PreUpdate();
    prob_->ResetCostEvolution(GetNumberOfMaxIterations() + 1);

    solution.resize(prob_->GetT(), prob_->N);
    solution.row(0) = prob_->GetInitialTrajectory()[0];
    int iter, feval, geval, ret;
    double f_sol;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        std::shared_ptr<OPTPP::OptQNewton> solver;
        if (parameters_.UseFiniteDifferences)
        {
            auto nlf_local = UnconstrainedTimeIndexedProblemWrapper(prob_).getFDNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptQNewton(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        else
        {
            auto nlf_local = UnconstrainedTimeIndexedProblemWrapper(prob_).getNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptQNewton(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        solver->setGradTol(parameters_.GradientTolerance);
        solver->setMaxBacktrackIter(parameters_.MaxBacktrackIterations);
        solver->setLineSearchTol(parameters_.LineSearchTolerance);
        solver->setStepTol(parameters_.StepTolerance);
        solver->setMaxIter(GetNumberOfMaxIterations());
        solver->setFcnTol(parameters_.FunctionTolerance);
        solver->setMinStep(parameters_.MinStep);
        solver->setOutputFile("/tmp/OPTPP_DEFAULT.out", 0);
        solver->optimize();
        ColumnVector sol = nlf->getXc();
        for (int t = 1; t < prob_->GetT(); t++)
            for (int i = 0; i < prob_->N; i++)
                solution(t, i) = sol((t - 1) * prob_->N + i + 1);
        iter = solver->getIter();
        feval = nlf->getFevals();
        geval = nlf->getGevals();
        f_sol = nlf->getF();
        ret = solver->getReturnCode();
        solver->cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        ThrowPretty("OPT++ exception:" << BaseException::what());
    }

    planning_time_ = timer.GetDuration();

    if (debug_)
    {
        HIGHLIGHT_NAMED(object_name_ + " OptppTrajQNewton", "Time: " << planning_time_ << ", Status: " << ret << ", Iterations: " << iter << ", Feval: " << feval << ", Geval: " << geval << ", Final cost: " << f_sol);
    }
}

template <>
void OptppTrajFDNewton::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) ThrowNamed("Solver has not been initialized!");
    prob_->PreUpdate();
    prob_->ResetCostEvolution(GetNumberOfMaxIterations() + 1);

    solution.resize(prob_->GetT(), prob_->N);
    solution.row(0) = prob_->GetInitialTrajectory()[0];
    int iter, feval, geval, ret;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        std::shared_ptr<OPTPP::OptFDNewton> solver(new OPTPP::OptFDNewton());
        if (parameters_.UseFiniteDifferences)
        {
            auto nlf_local = UnconstrainedTimeIndexedProblemWrapper(prob_).getFDNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptFDNewton(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        else
        {
            auto nlf_local = UnconstrainedTimeIndexedProblemWrapper(prob_).getNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptFDNewton(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        solver->setGradTol(parameters_.GradientTolerance);
        solver->setMaxBacktrackIter(parameters_.MaxBacktrackIterations);
        solver->setLineSearchTol(parameters_.LineSearchTolerance);
        solver->setStepTol(parameters_.StepTolerance);
        solver->setMaxIter(GetNumberOfMaxIterations());
        solver->setFcnTol(parameters_.FunctionTolerance);
        solver->setMinStep(parameters_.MinStep);
        solver->setOutputFile("/tmp/OPTPP_DEFAULT.out", 0);
        solver->optimize();
        ColumnVector sol = nlf->getXc();
        for (int t = 1; t < prob_->GetT(); t++)
            for (int i = 0; i < prob_->N; i++)
                solution(t, i) = sol((t - 1) * prob_->N + i + 1);
        iter = solver->getIter();
        feval = nlf->getFevals();
        geval = nlf->getGevals();
        ret = solver->getReturnCode();
        solver->cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        ThrowPretty("OPT++ exception:" << BaseException::what());
    }

    planning_time_ = timer.GetDuration();

    if (debug_)
    {
        HIGHLIGHT_NAMED(object_name_ + " OptppTrajFDNewton", "Time: " << planning_time_ << ", Status: " << ret << ", Iterations: " << iter << ", Feval: " << feval << ", Geval: " << geval);
    }
}

template <>
void OptppTrajGSS::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) ThrowNamed("Solver has not been initialized!");
    prob_->PreUpdate();
    prob_->ResetCostEvolution(GetNumberOfMaxIterations() + 1);

    solution.resize(prob_->GetT(), prob_->N);
    solution.row(0) = prob_->GetInitialTrajectory()[0];
    int iter, feval, geval, ret;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        std::shared_ptr<OPTPP::OptGSS> solver(new OPTPP::OptGSS());
        GenSetStd setBase(problem_->N);

        auto nlf_local = UnconstrainedTimeIndexedProblemWrapper(prob_).getFDNLF1();
        nlf = std::static_pointer_cast<NLP1>(nlf_local);
        solver.reset(new OPTPP::OptGSS(nlf.get(), &setBase));
        nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));

        solver->setOutputFile("/tmp/OPTPP_DEFAULT.out", 0);
        solver->setFullSearch(true);
        solver->setMaxIter(GetNumberOfMaxIterations());
        solver->optimize();
        ColumnVector sol = nlf->getXc();
        for (int t = 1; t < prob_->GetT(); t++)
            for (int i = 0; i < prob_->N; i++)
                solution(t, i) = sol((t - 1) * prob_->N + i + 1);
        iter = solver->getIter();
        feval = nlf->getFevals();
        ret = solver->getReturnCode();
        solver->cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        ThrowPretty("OPT++ exception:" << BaseException::what());
    }

    planning_time_ = timer.GetDuration();

    if (debug_)
    {
        HIGHLIGHT_NAMED(object_name_ + " OptppTrajGSS", "Time: " << planning_time_ << ", Status: " << ret << ", Iterations: " << iter << ", Feval: " << feval);
    }
}
}  // namespace exotica
