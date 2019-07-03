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

#ifndef EXOTICA_OPTPP_SOLVER_OPTPP_TRAJ_H_
#define EXOTICA_OPTPP_SOLVER_OPTPP_TRAJ_H_

#include <exotica_core/motion_solver.h>
#include <exotica_core/problems/unconstrained_time_indexed_problem.h>

#include <optpp_solver/OptppTrajCG_initializer.h>
#include <optpp_solver/OptppTrajFDNewton_initializer.h>
#include <optpp_solver/OptppTrajGSS_initializer.h>
#include <optpp_solver/OptppTrajLBFGS_initializer.h>
#include <optpp_solver/OptppTrajQNewton_initializer.h>
#include <optpp_solver/optpp_core.h>

namespace exotica
{
template <class ProblemType, class InitializerType>
class OptppTimeIndexedSolver : public MotionSolver, public Instantiable<InitializerType>
{
public:
    void Solve(Eigen::MatrixXd& solution) override;
    void SpecifyProblem(PlanningProblemPtr pointer) override;

private:
    std::shared_ptr<ProblemType> prob_;  // Shared pointer to the planning problem.
};

/// \brief LBFGS unconstrained trajectory solver
typedef OptppTimeIndexedSolver<UnconstrainedTimeIndexedProblem, OptppTrajLBFGSInitializer> OptppTrajLBFGS;

/// \brief Conjugate Gradient unconstrained trajectory solver
typedef OptppTimeIndexedSolver<UnconstrainedTimeIndexedProblem, OptppTrajCGInitializer> OptppTrajCG;

/// \brief Newton method unconstrained trajectory solver
typedef OptppTimeIndexedSolver<UnconstrainedTimeIndexedProblem, OptppTrajQNewtonInitializer> OptppTrajQNewton;

/// \brief Newton method unconstrained trajectory solver using finite differences for estimating the Hessian
typedef OptppTimeIndexedSolver<UnconstrainedTimeIndexedProblem, OptppTrajFDNewtonInitializer> OptppTrajFDNewton;

/// \brief Generating set search method unconstrained trajectory solver
typedef OptppTimeIndexedSolver<UnconstrainedTimeIndexedProblem, OptppTrajGSSInitializer> OptppTrajGSS;
}  // namespace exotica
#endif  // EXOTICA_OPTPP_SOLVER_OPTPP_TRAJ_H_
