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

#include <optpp_solver/otpp_traj.h>
#include <optpp_catkin/OptLBFGS.h>
#include <optpp_catkin/OptCG.h>
#include <optpp_catkin/OptQNewton.h>
#include <optpp_catkin/OptFDNewton.h>
#include <optpp_catkin/OptGSS.h>

REGISTER_MOTIONSOLVER_TYPE("OptppTrajLBFGS", exotica::OptppTrajLBFGS)
REGISTER_MOTIONSOLVER_TYPE("OptppTrajCG", exotica::OptppTrajCG)
REGISTER_MOTIONSOLVER_TYPE("OptppTrajQNewton", exotica::OptppTrajQNewton)
REGISTER_MOTIONSOLVER_TYPE("OptppTrajFDNewton", exotica::OptppTrajFDNewton)
REGISTER_MOTIONSOLVER_TYPE("OptppTrajGSS", exotica::OptppTrajGSS)

namespace exotica
{

void OptppTrajLBFGS::specifyProblem(PlanningProblem_ptr pointer)
{
    if (pointer->type() != "exotica::UnconstrainedTimeIndexedProblem")
    {
        throw_named("OPT++ IK can't solve problem of type '" << pointer->type() << "'!");
    }
    MotionSolver::specifyProblem(pointer);
    prob_ = std::static_pointer_cast<UnconstrainedTimeIndexedProblem>(pointer);
}

void OptppTrajLBFGS::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) throw_named("Solver has not been initialized!");
    prob_->preupdate();

    solution.resize(prob_->T, prob_->N);
    solution.row(0) = prob_->getInitialTrajectory()[0];
    int iter, feval, geval, ret;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        if(parameters_.UseFiniteDifferences)
        {
            nlf = std::static_pointer_cast<NLP1>(UnconstrainedTimeIndexedProblemWrapper(prob_).getFDNLF1());
        }
        else
        {
            nlf = std::static_pointer_cast<NLP1>(UnconstrainedTimeIndexedProblemWrapper(prob_).getNLF1());
        }
        OPTPP::OptLBFGS solver(nlf.get());
        solver.setGradTol(parameters_.GradientTolerance);
        solver.setMaxBacktrackIter(parameters_.MaxBacktrackIterations);
        solver.setLineSearchTol(parameters_.LineSearchTolerance);
        solver.setStepTol(parameters_.StepTolerance);
        solver.setMaxIter(parameters_.MaxIterations);
        solver.optimize();
        ColumnVector sol = nlf->getXc();
        for(int t=1; t<prob_->T; t++)
            for(int i=0; i<prob_->N; i++)
                solution(t,i) = sol((t-1)*prob_->N+i+1);
        iter = solver.getIter();
        feval = nlf->getFevals();
        geval = nlf->getGevals();
        ret = solver.getReturnCode();
        solver.cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        throw_pretty("OPT++ exception:"<<BaseException::what());
    }

    planning_time_ = timer.getDuration();

    if(debug_)
    {
        HIGHLIGHT_NAMED(object_name_+" OptppTrajLBFGS", "Time: "<<planning_time_<<" ,Status: "<<ret<<" , Iterations: "<<iter<<" ,Feval: "<<feval<<" , Geval: "<<geval);
    }
}



void OptppTrajCG::specifyProblem(PlanningProblem_ptr pointer)
{
    if (pointer->type() != "exotica::UnconstrainedTimeIndexedProblem")
    {
        throw_named("OPT++ IK can't solve problem of type '" << pointer->type() << "'!");
    }
    MotionSolver::specifyProblem(pointer);
    prob_ = std::static_pointer_cast<UnconstrainedTimeIndexedProblem>(pointer);
}

void OptppTrajCG::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) throw_named("Solver has not been initialized!");
    prob_->preupdate();

    solution.resize(prob_->T, prob_->N);
    solution.row(0) = prob_->getInitialTrajectory()[0];
    int iter, feval, geval, ret;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        if(parameters_.UseFiniteDifferences)
        {
            nlf = std::static_pointer_cast<NLP1>(UnconstrainedTimeIndexedProblemWrapper(prob_).getFDNLF1());
        }
        else
        {
            nlf = std::static_pointer_cast<NLP1>(UnconstrainedTimeIndexedProblemWrapper(prob_).getNLF1());
        }
        OPTPP::OptCG solver(nlf.get());
        solver.setGradTol(parameters_.GradientTolerance);
        solver.setMaxBacktrackIter(parameters_.MaxBacktrackIterations);
        solver.setLineSearchTol(parameters_.LineSearchTolerance);
        solver.setMaxIter(parameters_.MaxIterations);
        solver.optimize();
        ColumnVector sol = nlf->getXc();
        for(int t=1; t<prob_->T; t++)
            for(int i=0; i<prob_->N; i++)
                solution(t,i) = sol((t-1)*prob_->N+i+1);
        iter = solver.getIter();
        feval = nlf->getFevals();
        geval = nlf->getGevals();
        ret = solver.getReturnCode();
        solver.cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        throw_pretty("OPT++ exception:"<<BaseException::what());
    }

    planning_time_ = timer.getDuration();

    if(debug_)
    {
        HIGHLIGHT_NAMED(object_name_+" OptppTrajCG", "Time: "<<planning_time_<<" ,Status: "<<ret<<" , Iterations: "<<iter<<" ,Feval: "<<feval<<" , Geval: "<<geval);
    }
}




void OptppTrajQNewton::specifyProblem(PlanningProblem_ptr pointer)
{
    if (pointer->type() != "exotica::UnconstrainedTimeIndexedProblem")
    {
        throw_named("OPT++ IK can't solve problem of type '" << pointer->type() << "'!");
    }
    MotionSolver::specifyProblem(pointer);
    prob_ = std::static_pointer_cast<UnconstrainedTimeIndexedProblem>(pointer);
}

void OptppTrajQNewton::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) throw_named("Solver has not been initialized!");
    prob_->preupdate();

    solution.resize(prob_->T, prob_->N);
    solution.row(0) = prob_->getInitialTrajectory()[0];
    int iter, feval, geval, ret;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        if(parameters_.UseFiniteDifferences)
        {
            nlf = std::static_pointer_cast<NLP1>(UnconstrainedTimeIndexedProblemWrapper(prob_).getFDNLF1());
        }
        else
        {
            nlf = std::static_pointer_cast<NLP1>(UnconstrainedTimeIndexedProblemWrapper(prob_).getNLF1());
        }
        OPTPP::OptQNewton solver(nlf.get());
        solver.setGradTol(parameters_.GradientTolerance);
        solver.setMaxBacktrackIter(parameters_.MaxBacktrackIterations);
        solver.setLineSearchTol(parameters_.LineSearchTolerance);
        solver.setMaxIter(parameters_.MaxIterations);
        solver.optimize();
        ColumnVector sol = nlf->getXc();
        for(int t=1; t<prob_->T; t++)
            for(int i=0; i<prob_->N; i++)
                solution(t,i) = sol((t-1)*prob_->N+i+1);
        iter = solver.getIter();
        feval = nlf->getFevals();
        geval = nlf->getGevals();
        ret = solver.getReturnCode();
        solver.cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        throw_pretty("OPT++ exception:"<<BaseException::what());
    }

    planning_time_ = timer.getDuration();

    if(debug_)
    {
        HIGHLIGHT_NAMED(object_name_+" OptppTrajQNewton", "Time: "<<planning_time_<<" ,Status: "<<ret<<" , Iterations: "<<iter<<" ,Feval: "<<feval<<" , Geval: "<<geval);
    }
}





void OptppTrajFDNewton::specifyProblem(PlanningProblem_ptr pointer)
{
    if (pointer->type() != "exotica::UnconstrainedTimeIndexedProblem")
    {
        throw_named("OPT++ IK can't solve problem of type '" << pointer->type() << "'!");
    }
    MotionSolver::specifyProblem(pointer);
    prob_ = std::static_pointer_cast<UnconstrainedTimeIndexedProblem>(pointer);
}

void OptppTrajFDNewton::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) throw_named("Solver has not been initialized!");
    prob_->preupdate();

    solution.resize(prob_->T, prob_->N);
    solution.row(0) = prob_->getInitialTrajectory()[0];
    int iter, feval, geval, ret;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        if(parameters_.UseFiniteDifferences)
        {
            nlf = std::static_pointer_cast<NLP1>(UnconstrainedTimeIndexedProblemWrapper(prob_).getFDNLF1());
        }
        else
        {
            nlf = std::static_pointer_cast<NLP1>(UnconstrainedTimeIndexedProblemWrapper(prob_).getNLF1());
        }
        OPTPP::OptFDNewton solver(nlf.get());
        solver.setGradTol(parameters_.GradientTolerance);
        solver.setMaxBacktrackIter(parameters_.MaxBacktrackIterations);
        solver.setLineSearchTol(parameters_.LineSearchTolerance);
        solver.setMaxIter(parameters_.MaxIterations);
        solver.optimize();
        ColumnVector sol = nlf->getXc();
        for(int t=1; t<prob_->T; t++)
            for(int i=0; i<prob_->N; i++)
                solution(t,i) = sol((t-1)*prob_->N+i+1);
        iter = solver.getIter();
        feval = nlf->getFevals();
        geval = nlf->getGevals();
        ret = solver.getReturnCode();
        solver.cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        throw_pretty("OPT++ exception:"<<BaseException::what());
    }

    planning_time_ = timer.getDuration();

    if(debug_)
    {
        HIGHLIGHT_NAMED(object_name_+" OptppTrajFDNewton", "Time: "<<planning_time_<<" ,Status: "<<ret<<" , Iterations: "<<iter<<" ,Feval: "<<feval<<" , Geval: "<<geval);
    }
}







void OptppTrajGSS::specifyProblem(PlanningProblem_ptr pointer)
{
    if (pointer->type() != "exotica::UnconstrainedTimeIndexedProblem")
    {
        throw_named("OPT++ IK can't solve problem of type '" << pointer->type() << "'!");
    }
    MotionSolver::specifyProblem(pointer);
    prob_ = std::static_pointer_cast<UnconstrainedTimeIndexedProblem>(pointer);
}

void OptppTrajGSS::Solve(Eigen::MatrixXd& solution)
{
    Timer timer;

    if (!prob_) throw_named("Solver has not been initialized!");
    prob_->preupdate();

    solution.resize(prob_->T, prob_->N);
    solution.row(0) = prob_->getInitialTrajectory()[0];
    int iter, feval, geval, ret;

    Try
    {
        std::shared_ptr<NLP1> nlf;
        nlf = std::static_pointer_cast<NLP1>(UnconstrainedTimeIndexedProblemWrapper(prob_).getFDNLF1());
        GenSetStd setBase(problem_->N);
        OPTPP::OptGSS solver(nlf.get(), &setBase);
        solver.setFullSearch(true);
        solver.setMaxIter(parameters_.MaxIterations);
        solver.optimize();
        ColumnVector sol = nlf->getXc();
        for(int t=1; t<prob_->T; t++)
            for(int i=0; i<prob_->N; i++)
                solution(t,i) = sol((t-1)*prob_->N+i+1);
        iter = solver.getIter();
        feval = nlf->getFevals();
        ret = solver.getReturnCode();
        solver.cleanup();
    }
    CatchAll
    {
        Tracer::last->PrintTrace();
        throw_pretty("OPT++ exception:"<<BaseException::what());
    }

    planning_time_ = timer.getDuration();

    if(debug_)
    {
        HIGHLIGHT_NAMED(object_name_+" OptppTrajGSS", "Time: "<<planning_time_<<" ,Status: "<<ret<<" , Iterations: "<<iter<<" ,Feval: "<<feval);
    }
}

}
