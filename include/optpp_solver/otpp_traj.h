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

#ifndef OPTPPTRAJ_H
#define OPTPPTRAJ_H

#include <exotica/Exotica.h>
#include <exotica/Problems/UnconstrainedTimeIndexedProblem.h>
#include <optpp_solver/OptppTrajCGInitializer.h>
#include <optpp_solver/OptppTrajFDNewtonInitializer.h>
#include <optpp_solver/OptppTrajGSSInitializer.h>
#include <optpp_solver/OptppTrajLBFGSInitializer.h>
#include <optpp_solver/OptppTrajQNewtonInitializer.h>
#include <optpp_solver/optpp_core.h>

namespace exotica
{
/// \brief LBFGS IK solver
class OptppTrajLBFGS : public MotionSolver, public Instantiable<OptppTrajLBFGSInitializer>
{
public:
    OptppTrajLBFGS() {}
    virtual ~OptppTrajLBFGS() {}
    virtual void Instantiate(OptppTrajLBFGSInitializer& init)
    {
        parameters_ = init;
        setNumberOfMaxIterations(parameters_.MaxIterations);
    }
    virtual void Solve(Eigen::MatrixXd& solution);

    virtual void specifyProblem(PlanningProblem_ptr pointer);

    UnconstrainedTimeIndexedProblem_ptr& getProblem() { return prob_; }
private:
    OptppTrajLBFGSInitializer parameters_;

    UnconstrainedTimeIndexedProblem_ptr prob_;  // Shared pointer to the planning problem.
};
typedef std::shared_ptr<exotica::OptppTrajLBFGS> OptppTrajLBFGS_ptr;

/// \brief Conjugate Gradient IK solver
class OptppTrajCG : public MotionSolver, public Instantiable<OptppTrajCGInitializer>
{
public:
    OptppTrajCG() {}
    virtual ~OptppTrajCG() {}
    virtual void Instantiate(OptppTrajCGInitializer& init)
    {
        parameters_ = init;
        setNumberOfMaxIterations(parameters_.MaxIterations);
    }
    virtual void Solve(Eigen::MatrixXd& solution);

    virtual void specifyProblem(PlanningProblem_ptr pointer);

    UnconstrainedTimeIndexedProblem_ptr& getProblem() { return prob_; }
private:
    OptppTrajCGInitializer parameters_;

    UnconstrainedTimeIndexedProblem_ptr prob_;  // Shared pointer to the planning problem.
};
typedef std::shared_ptr<exotica::OptppTrajCG> OptppTrajCG_ptr;

/// \brief Newton method IK solver
class OptppTrajQNewton : public MotionSolver, public Instantiable<OptppTrajQNewtonInitializer>
{
public:
    OptppTrajQNewton() {}
    virtual ~OptppTrajQNewton() {}
    virtual void Instantiate(OptppTrajQNewtonInitializer& init)
    {
        parameters_ = init;
        setNumberOfMaxIterations(parameters_.MaxIterations);
    }
    virtual void Solve(Eigen::MatrixXd& solution);

    virtual void specifyProblem(PlanningProblem_ptr pointer);

    UnconstrainedTimeIndexedProblem_ptr& getProblem() { return prob_; }
private:
    OptppTrajQNewtonInitializer parameters_;

    UnconstrainedTimeIndexedProblem_ptr prob_;  // Shared pointer to the planning problem.
};
typedef std::shared_ptr<exotica::OptppTrajQNewton> OptppTrajQNewton_ptr;

/// \brief Newton method IK solver using finite differences for estimating the Hessian
class OptppTrajFDNewton : public MotionSolver, public Instantiable<OptppTrajFDNewtonInitializer>
{
public:
    OptppTrajFDNewton() {}
    virtual ~OptppTrajFDNewton() {}
    virtual void Instantiate(OptppTrajFDNewtonInitializer& init)
    {
        parameters_ = init;
        setNumberOfMaxIterations(parameters_.MaxIterations);
    }
    virtual void Solve(Eigen::MatrixXd& solution);

    virtual void specifyProblem(PlanningProblem_ptr pointer);

    UnconstrainedTimeIndexedProblem_ptr& getProblem() { return prob_; }
private:
    OptppTrajFDNewtonInitializer parameters_;

    UnconstrainedTimeIndexedProblem_ptr prob_;  // Shared pointer to the planning problem.
};
typedef std::shared_ptr<exotica::OptppTrajFDNewton> OptppTrajFDNewton_ptr;

/// \brief Generating set search method IK solver
class OptppTrajGSS : public MotionSolver, public Instantiable<OptppTrajGSSInitializer>
{
public:
    OptppTrajGSS() {}
    virtual ~OptppTrajGSS() {}
    virtual void Instantiate(OptppTrajGSSInitializer& init)
    {
        parameters_ = init;
        setNumberOfMaxIterations(parameters_.MaxIterations);
    }
    virtual void Solve(Eigen::MatrixXd& solution);

    virtual void specifyProblem(PlanningProblem_ptr pointer);

    UnconstrainedTimeIndexedProblem_ptr& getProblem() { return prob_; }
private:
    OptppTrajGSSInitializer parameters_;

    UnconstrainedTimeIndexedProblem_ptr prob_;  // Shared pointer to the planning problem.
};
typedef std::shared_ptr<exotica::OptppTrajGSS> OptppTrajGSS_ptr;
}
#endif
