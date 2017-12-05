/*
 *  Created on: 19 Oct 2017
 *      Author: Vladimir Ivan
 *
 *  This code is based on algorithm developed by Marc Toussaint
 *  M. Toussaint: Robot Trajectory Optimization using Approximate Inference. In Proc. of the Int. Conf. on Machine Learning (ICML 2009), 1049-1056, ACM, 2009.
 *  http://ipvs.informatik.uni-stuttgart.de/mlr/papers/09-toussaint-ICML.pdf
 *  Original code available at http://ipvs.informatik.uni-stuttgart.de/mlr/marc/source-code/index.html
 *
 * Copyright (c) 2017, University Of Edinburgh
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of  nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <optpp_catkin/OptCG.h>
#include <optpp_catkin/OptFDNewton.h>
#include <optpp_catkin/OptGSS.h>
#include <optpp_catkin/OptLBFGS.h>
#include <optpp_catkin/OptQNewton.h>
#include <optpp_solver/otpp_ik.h>

REGISTER_MOTIONSOLVER_TYPE("OptppIKLBFGS", exotica::OptppIKLBFGS)
REGISTER_MOTIONSOLVER_TYPE("OptppIKCG", exotica::OptppIKCG)
REGISTER_MOTIONSOLVER_TYPE("OptppIKQNewton", exotica::OptppIKQNewton)
REGISTER_MOTIONSOLVER_TYPE("OptppIKFDNewton", exotica::OptppIKFDNewton)
REGISTER_MOTIONSOLVER_TYPE("OptppIKGSS", exotica::OptppIKGSS)

namespace exotica
{
void OptppIKLBFGS::specifyProblem(PlanningProblem_ptr pointer)
{
    if (pointer->type() != "exotica::UnconstrainedEndPoseProblem")
    {
        throw_named("OPT++ IK can't solve problem of type '" << pointer->type() << "'!");
    }
    MotionSolver::specifyProblem(pointer);
    prob_ = std::static_pointer_cast<UnconstrainedEndPoseProblem>(pointer);
}

void OptppIKLBFGS::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) throw_named("Solver has not been initialized!");
    prob_->preupdate();
    prob_->resetCostEvolution(parameters_.MaxIterations);

    solution.resize(1, prob_->N);
    int iter, feval, geval, ret;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        std::shared_ptr<OPTPP::OptLBFGS> solver(new OPTPP::OptLBFGS());
        if (parameters_.UseFiniteDifferences)
        {
            auto nlf_local = UnconstrainedEndPoseProblemWrapper(prob_).getFDNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptLBFGS(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        else
        {
            auto nlf_local = UnconstrainedEndPoseProblemWrapper(prob_).getNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptLBFGS(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        solver->setGradTol(parameters_.GradientTolerance);
        solver->setMaxBacktrackIter(parameters_.MaxBacktrackIterations);
        solver->setLineSearchTol(parameters_.LineSearchTolerance);
        solver->setMaxIter(parameters_.MaxIterations);
        ColumnVector W(prob_->N);
        for (int i = 0; i < prob_->N; i++) W(i + 1) = prob_->W(i, i);
        solver->setXScale(W);
        solver->optimize();
        ColumnVector sol = nlf->getXc();
        for (int i = 0; i < prob_->N; i++) solution(0, i) = sol(i + 1);
        iter = solver->getIter();
        feval = nlf->getFevals();
        geval = nlf->getGevals();
        ret = solver->getReturnCode();
        solver->cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        throw_pretty("OPT++ exception:" << BaseException::what());
    }

    planning_time_ = timer.getDuration();

    if (debug_)
    {
        HIGHLIGHT_NAMED(object_name_ + " OptppIKLBFGS", "Time: " << planning_time_ << " ,Status: " << ret << " , Iterations: " << iter << " ,Feval: " << feval << " , Geval: " << geval);
    }
}

void OptppIKCG::specifyProblem(PlanningProblem_ptr pointer)
{
    if (pointer->type() != "exotica::UnconstrainedEndPoseProblem")
    {
        throw_named("OPT++ IK can't solve problem of type '" << pointer->type() << "'!");
    }
    MotionSolver::specifyProblem(pointer);
    prob_ = std::static_pointer_cast<UnconstrainedEndPoseProblem>(pointer);
}

void OptppIKCG::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) throw_named("Solver has not been initialized!");
    prob_->preupdate();
    prob_->resetCostEvolution(parameters_.MaxIterations);

    solution.resize(1, prob_->N);
    int iter, feval, geval, ret;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        std::shared_ptr<OPTPP::OptCG> solver(new OPTPP::OptCG());
        if (parameters_.UseFiniteDifferences)
        {
            auto nlf_local = UnconstrainedEndPoseProblemWrapper(prob_).getFDNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptCG(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        else
        {
            auto nlf_local = UnconstrainedEndPoseProblemWrapper(prob_).getNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptCG(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        solver->setGradTol(parameters_.GradientTolerance);
        solver->setMaxBacktrackIter(parameters_.MaxBacktrackIterations);
        solver->setLineSearchTol(parameters_.LineSearchTolerance);
        solver->setMaxIter(parameters_.MaxIterations);
        ColumnVector W(prob_->N);
        for (int i = 0; i < prob_->N; i++) W(i + 1) = prob_->W(i, i);
        solver->setXScale(W);
        solver->optimize();
        ColumnVector sol = nlf->getXc();
        for (int i = 0; i < prob_->N; i++) solution(0, i) = sol(i + 1);
        iter = solver->getIter();
        feval = nlf->getFevals();
        geval = nlf->getGevals();
        ret = solver->getReturnCode();
        solver->cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        throw_pretty("OPT++ exception:" << BaseException::what());
    }

    planning_time_ = timer.getDuration();

    if (debug_)
    {
        HIGHLIGHT_NAMED(object_name_ + " OptppIKCG", "Time: " << planning_time_ << " ,Status: " << ret << " , Iterations: " << iter << " ,Feval: " << feval << " , Geval: " << geval);
    }
}

void OptppIKQNewton::specifyProblem(PlanningProblem_ptr pointer)
{
    if (pointer->type() != "exotica::UnconstrainedEndPoseProblem")
    {
        throw_named("OPT++ IK can't solve problem of type '" << pointer->type() << "'!");
    }
    MotionSolver::specifyProblem(pointer);
    prob_ = std::static_pointer_cast<UnconstrainedEndPoseProblem>(pointer);
}

void OptppIKQNewton::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) throw_named("Solver has not been initialized!");
    prob_->preupdate();
    prob_->resetCostEvolution(parameters_.MaxIterations);

    solution.resize(1, prob_->N);
    int iter, feval, geval, ret;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        std::shared_ptr<OPTPP::OptQNewton> solver(new OPTPP::OptQNewton());
        if (parameters_.UseFiniteDifferences)
        {
            auto nlf_local = UnconstrainedEndPoseProblemWrapper(prob_).getFDNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptQNewton(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        else
        {
            auto nlf_local = UnconstrainedEndPoseProblemWrapper(prob_).getNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptQNewton(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        solver->setGradTol(parameters_.GradientTolerance);
        solver->setMaxBacktrackIter(parameters_.MaxBacktrackIterations);
        solver->setLineSearchTol(parameters_.LineSearchTolerance);
        solver->setMaxIter(parameters_.MaxIterations);
        ColumnVector W(prob_->N);
        for (int i = 0; i < prob_->N; i++) W(i + 1) = prob_->W(i, i);
        solver->setXScale(W);
        solver->optimize();
        ColumnVector sol = nlf->getXc();
        for (int i = 0; i < prob_->N; i++) solution(0, i) = sol(i + 1);
        iter = solver->getIter();
        feval = nlf->getFevals();
        geval = nlf->getGevals();
        ret = solver->getReturnCode();
        solver->cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        throw_pretty("OPT++ exception:" << BaseException::what());
    }

    planning_time_ = timer.getDuration();

    if (debug_)
    {
        HIGHLIGHT_NAMED(object_name_ + " OptppIKQNewton", "Time: " << planning_time_ << " ,Status: " << ret << " , Iterations: " << iter << " ,Feval: " << feval << " , Geval: " << geval);
    }
}

void OptppIKFDNewton::specifyProblem(PlanningProblem_ptr pointer)
{
    if (pointer->type() != "exotica::UnconstrainedEndPoseProblem")
    {
        throw_named("OPT++ IK can't solve problem of type '" << pointer->type() << "'!");
    }
    MotionSolver::specifyProblem(pointer);
    prob_ = std::static_pointer_cast<UnconstrainedEndPoseProblem>(pointer);
}

void OptppIKFDNewton::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) throw_named("Solver has not been initialized!");
    prob_->preupdate();
    prob_->resetCostEvolution(parameters_.MaxIterations);

    solution.resize(1, prob_->N);
    int iter, feval, geval, ret;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        std::shared_ptr<OPTPP::OptFDNewton> solver(new OPTPP::OptFDNewton());
        if (parameters_.UseFiniteDifferences)
        {
            auto nlf_local = UnconstrainedEndPoseProblemWrapper(prob_).getFDNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptFDNewton(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        else
        {
            auto nlf_local = UnconstrainedEndPoseProblemWrapper(prob_).getNLF1();
            nlf = std::static_pointer_cast<NLP1>(nlf_local);
            solver.reset(new OPTPP::OptFDNewton(nlf.get()));
            nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));
        }
        solver->setGradTol(parameters_.GradientTolerance);
        solver->setMaxBacktrackIter(parameters_.MaxBacktrackIterations);
        solver->setLineSearchTol(parameters_.LineSearchTolerance);
        solver->setMaxIter(parameters_.MaxIterations);
        ColumnVector W(prob_->N);
        for (int i = 0; i < prob_->N; i++) W(i + 1) = prob_->W(i, i);
        solver->setXScale(W);
        solver->optimize();
        ColumnVector sol = nlf->getXc();
        for (int i = 0; i < prob_->N; i++) solution(0, i) = sol(i + 1);
        iter = solver->getIter();
        feval = nlf->getFevals();
        geval = nlf->getGevals();
        ret = solver->getReturnCode();
        solver->cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        throw_pretty("OPT++ exception:" << BaseException::what());
    }

    planning_time_ = timer.getDuration();

    if (debug_)
    {
        HIGHLIGHT_NAMED(object_name_ + " OptppIKFDNewton", "Time: " << planning_time_ << " ,Status: " << ret << " , Iterations: " << iter << " ,Feval: " << feval << " , Geval: " << geval);
    }
}

void OptppIKGSS::specifyProblem(PlanningProblem_ptr pointer)
{
    if (pointer->type() != "exotica::UnconstrainedEndPoseProblem")
    {
        throw_named("OPT++ IK can't solve problem of type '" << pointer->type() << "'!");
    }
    MotionSolver::specifyProblem(pointer);
    prob_ = std::static_pointer_cast<UnconstrainedEndPoseProblem>(pointer);
}

void OptppIKGSS::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) throw_named("Solver has not been initialized!");
    prob_->preupdate();
    prob_->resetCostEvolution(parameters_.MaxIterations);

    solution.resize(1, prob_->N);
    int iter, feval, geval, ret;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        std::shared_ptr<OPTPP::OptGSS> solver(new OPTPP::OptGSS());
        GenSetStd setBase(problem_->N);

        auto nlf_local = UnconstrainedEndPoseProblemWrapper(prob_).getFDNLF1();
        nlf = std::static_pointer_cast<NLP1>(nlf_local);
        solver.reset(new OPTPP::OptGSS(nlf.get(), &setBase));
        nlf_local->setSolver(std::static_pointer_cast<OPTPP::OptimizeClass>(solver));

        solver->setFullSearch(true);
        solver->setMaxIter(parameters_.MaxIterations);
        ColumnVector W(prob_->N);
        for (int i = 0; i < prob_->N; i++) W(i + 1) = prob_->W(i, i);
        solver->setXScale(W);
        solver->optimize();
        ColumnVector sol = nlf->getXc();
        for (int i = 0; i < prob_->N; i++) solution(0, i) = sol(i + 1);
        iter = solver->getIter();
        feval = nlf->getFevals();
        ret = solver->getReturnCode();
        solver->cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        throw_pretty("OPT++ exception:" << BaseException::what());
    }

    planning_time_ = timer.getDuration();

    if (debug_)
    {
        HIGHLIGHT_NAMED(object_name_ + " OptppIKGSS", "Time: " << planning_time_ << " ,Status: " << ret << " , Iterations: " << iter << " ,Feval: " << feval);
    }
}
}
