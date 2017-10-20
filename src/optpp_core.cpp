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

#include <optpp_solver/optpp_core.h>
#include <Eigen/Dense>

namespace exotica
{

UnconstrainedEndPoseProblemWrapper::UnconstrainedEndPoseProblemWrapper(UnconstrainedEndPoseProblem_ptr problem) :
    problem_(problem), n_(problem_->getScene()->getSolver().getNumControlledJoints())
{
    if(problem_->getNominalPose().rows()>0) throw_pretty("OPT++ solvers don't support null-space optimization! "<<problem_->getNominalPose().rows());
}

void UnconstrainedEndPoseProblemWrapper::updateCallback(int mode, int n, const ColumnVector& x, double& fx, ColumnVector& gx, int& result, void* data)
{
    reinterpret_cast<UnconstrainedEndPoseProblemWrapper*>(data)->update(mode, n, x, fx, gx, result);
}

void UnconstrainedEndPoseProblemWrapper::updateCallbackFD(int n, const ColumnVector& x, double& fx, int& result, void* data)
{
    ColumnVector gx;
    reinterpret_cast<UnconstrainedEndPoseProblemWrapper*>(data)->update(NLPFunction, n, x, fx, gx, result);
}

void UnconstrainedEndPoseProblemWrapper::update(int mode, int n, const ColumnVector& x_opp, double& fx, ColumnVector& gx, int& result)
{
    if(n!=n_) throw_pretty("Invalid OPT++ state size, expecting "<<n_<<" got "<<n);
    Eigen::VectorXd x(n);
    for(int i=0; i<n; i++) x(i) = x_opp(i+1);
    problem_->Update(x);
    Eigen::VectorXd yd = problem_->Phi - problem_->y;

    if (mode & NLPFunction)
    {
        fx = problem_->getScalarCost();
        result = NLPFunction;
    }

    if (mode & NLPGradient)
    {
        Eigen::VectorXd J = problem_->getScalarJacobian();
        for(int i=0; i<n; i++) gx(i+1) = J(i);
        result = NLPGradient;
    }
}

void UnconstrainedEndPoseProblemWrapper::init(int n, ColumnVector& x)
{
    if(n!=n_) throw_pretty("Invalid OPT++ state size, expecting "<<n_<<" got "<<n);
    Eigen::VectorXd x0 = problem_->applyStartState();
    x.ReSize(n);
    for(int i=0; i<n; i++) x(i+1) = x0(i);
}

FDNLF1WrapperUEPP UnconstrainedEndPoseProblemWrapper::getFDNLF1()
{
    return FDNLF1WrapperUEPP(*this);
}

NLF1WrapperUEPP UnconstrainedEndPoseProblemWrapper::getNLF1()
{
    return NLF1WrapperUEPP(*this);
}

NLF1WrapperUEPP::NLF1WrapperUEPP(const UnconstrainedEndPoseProblemWrapper& parent) : parent_(parent),
    NLF1(parent.n_, UnconstrainedEndPoseProblemWrapper::updateCallback, nullptr, (void*)nullptr)
{
    vptr = reinterpret_cast<UnconstrainedEndPoseProblemWrapper*>(&parent_);
}

void NLF1WrapperUEPP::initFcn()
{
    if (init_flag == false)
    {
        parent_.init(dim, mem_xc);
        init_flag = true;
    }
    else
    {
      parent_.init(dim, mem_xc);
    }
}

FDNLF1WrapperUEPP::FDNLF1WrapperUEPP(const UnconstrainedEndPoseProblemWrapper& parent) : parent_(parent),
    FDNLF1(parent.n_, UnconstrainedEndPoseProblemWrapper::updateCallbackFD, nullptr, (void*)nullptr)
{
    vptr = reinterpret_cast<UnconstrainedEndPoseProblemWrapper*>(&parent_);
}

void FDNLF1WrapperUEPP::initFcn()
{
    if (init_flag == false)
    {
        parent_.init(dim, mem_xc);
        init_flag = true;
    }
    else
    {
      parent_.init(dim, mem_xc);
    }
}

}
