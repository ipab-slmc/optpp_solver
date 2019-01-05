//
// Copyright (c) 2018, University of Edinburgh
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

#ifndef EXOTICA_OPTPP_SOLVER_OPTPP_IK_H_
#define EXOTICA_OPTPP_SOLVER_OPTPP_IK_H_

#include <exotica_core/exotica_core.h>
#include <exotica_core/problems/unconstrained_end_pose_problem.h>
#include <optpp_solver/OptppIKCG_initializer.h>
#include <optpp_solver/OptppIKFDNewton_initializer.h>
#include <optpp_solver/OptppIKGSS_initializer.h>
#include <optpp_solver/OptppIKLBFGS_initializer.h>
#include <optpp_solver/OptppIKQNewton_initializer.h>
#include <optpp_solver/optpp_core.h>

namespace exotica
{
/// \brief LBFGS IK solver
class OptppIKLBFGS : public MotionSolver, public Instantiable<OptppIKLBFGSInitializer>
{
public:
    OptppIKLBFGS() {}
    virtual ~OptppIKLBFGS() {}
    virtual void Instantiate(OptppIKLBFGSInitializer& init)
    {
        parameters_ = init;
    }
    virtual void Solve(Eigen::MatrixXd& solution);

    virtual void SpecifyProblem(PlanningProblemPtr pointer);

    UnconstrainedEndPoseProblemPtr& GetProblem() { return prob_; }
private:
    OptppIKLBFGSInitializer parameters_;

    UnconstrainedEndPoseProblemPtr prob_;  // Shared pointer to the planning problem.
};

/// \brief Conjugate Gradient IK solver
class OptppIKCG : public MotionSolver, public Instantiable<OptppIKCGInitializer>
{
public:
    OptppIKCG() {}
    virtual ~OptppIKCG() {}
    virtual void Instantiate(OptppIKCGInitializer& init)
    {
        parameters_ = init;
    }
    virtual void Solve(Eigen::MatrixXd& solution);

    virtual void SpecifyProblem(PlanningProblemPtr pointer);

    UnconstrainedEndPoseProblemPtr& GetProblem() { return prob_; }
private:
    OptppIKCGInitializer parameters_;

    UnconstrainedEndPoseProblemPtr prob_;  // Shared pointer to the planning problem.
};

/// \brief Newton method IK solver
class OptppIKQNewton : public MotionSolver, public Instantiable<OptppIKQNewtonInitializer>
{
public:
    OptppIKQNewton() {}
    virtual ~OptppIKQNewton() {}
    virtual void Instantiate(OptppIKQNewtonInitializer& init)
    {
        parameters_ = init;
    }
    virtual void Solve(Eigen::MatrixXd& solution);

    virtual void SpecifyProblem(PlanningProblemPtr pointer);

    UnconstrainedEndPoseProblemPtr& GetProblem() { return prob_; }
private:
    OptppIKQNewtonInitializer parameters_;

    UnconstrainedEndPoseProblemPtr prob_;  // Shared pointer to the planning problem.
};

/// \brief Newton method IK solver using finite differences for estimating the Hessian
class OptppIKFDNewton : public MotionSolver, public Instantiable<OptppIKFDNewtonInitializer>
{
public:
    OptppIKFDNewton() {}
    virtual ~OptppIKFDNewton() {}
    virtual void Instantiate(OptppIKFDNewtonInitializer& init)
    {
        parameters_ = init;
    }
    virtual void Solve(Eigen::MatrixXd& solution);

    virtual void SpecifyProblem(PlanningProblemPtr pointer);

    UnconstrainedEndPoseProblemPtr& GetProblem() { return prob_; }
private:
    OptppIKFDNewtonInitializer parameters_;

    UnconstrainedEndPoseProblemPtr prob_;  // Shared pointer to the planning problem.
};

/// \brief Generating set search method IK solver
class OptppIKGSS : public MotionSolver, public Instantiable<OptppIKGSSInitializer>
{
public:
    OptppIKGSS() {}
    virtual ~OptppIKGSS() {}
    virtual void Instantiate(OptppIKGSSInitializer& init)
    {
        parameters_ = init;
    }
    virtual void Solve(Eigen::MatrixXd& solution);

    virtual void SpecifyProblem(PlanningProblemPtr pointer);

    UnconstrainedEndPoseProblemPtr& GetProblem() { return prob_; }
private:
    OptppIKGSSInitializer parameters_;

    UnconstrainedEndPoseProblemPtr prob_;  // Shared pointer to the planning problem.
};
}
#endif
