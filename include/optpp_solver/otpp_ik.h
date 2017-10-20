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

#ifndef OPTPPIK_H
#define OPTPPIK_H

#include <optpp_solver/optpp_core.h>
#include <exotica/Exotica.h>
#include <exotica/Problems/UnconstrainedEndPoseProblem.h>
#include <optpp_solver/OptppIKLBFGSInitializer.h>

namespace exotica
{
/// \brief	OPT++ IK solver

class OptppIKLBFGS : public MotionSolver, public Instantiable<OptppIKLBFGSInitializer>
{
public:
    OptppIKLBFGS() {}
    virtual ~OptppIKLBFGS() {}

    virtual void Instantiate(OptppIKLBFGSInitializer& init) { parameters_ = init;}

    virtual void Solve(Eigen::MatrixXd& solution);

    virtual void specifyProblem(PlanningProblem_ptr pointer);

    UnconstrainedEndPoseProblem_ptr& getProblem() { return prob_;}

    double planning_time_;

private:
    OptppIKLBFGSInitializer parameters_;

    UnconstrainedEndPoseProblem_ptr prob_;  // Shared pointer to the planning problem.
};
typedef std::shared_ptr<exotica::OptppIKLBFGS> OptppIKLBFGS_ptr;

}

#endif // OPTPPIK_H
