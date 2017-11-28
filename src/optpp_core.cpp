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
    problem_(problem), n_(problem_->N)
{
    if(problem_->getNominalPose().rows()>0) throw_pretty("OPT++ solvers don't support null-space optimization! "<<problem_->getNominalPose().rows());
}

void UnconstrainedEndPoseProblemWrapper::setSolver(std::shared_ptr<OPTPP::OptimizeClass> solver)
{
    solver_ = solver;
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

    // Store cost
    int iter = solver_->getIter();
    if (iter == 1) hasBeenInitialized = true;
    if (!hasBeenInitialized) iter = 0;
    // HIGHLIGHT_NAMED("UEPPW::update", "iter: " << iter << " cost: " << fx)
    problem_->setCostEvolution(iter, fx);
}

void UnconstrainedEndPoseProblemWrapper::init(int n, ColumnVector& x)
{
    if(n!=n_) throw_pretty("Invalid OPT++ state size, expecting "<<n_<<" got "<<n);
    Eigen::VectorXd x0 = problem_->applyStartState();
    x.ReSize(n);
    for(int i=0; i<n; i++) x(i+1) = x0(i);
    hasBeenInitialized = false;
}

std::shared_ptr<FDNLF1WrapperUEPP> UnconstrainedEndPoseProblemWrapper::getFDNLF1()
{
    return std::shared_ptr<FDNLF1WrapperUEPP>(new FDNLF1WrapperUEPP(*this));
}

std::shared_ptr<NLF1WrapperUEPP> UnconstrainedEndPoseProblemWrapper::getNLF1()
{
    return std::shared_ptr<NLF1WrapperUEPP>(new NLF1WrapperUEPP(*this));
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









UnconstrainedTimeIndexedProblemWrapper::UnconstrainedTimeIndexedProblemWrapper(UnconstrainedTimeIndexedProblem_ptr problem) :
    problem_(problem), n_(problem_->N*(problem_->T-1))
{

}

void UnconstrainedTimeIndexedProblemWrapper::setSolver(std::shared_ptr<OPTPP::OptimizeClass> solver)
{
    solver_ = solver;
}

void UnconstrainedTimeIndexedProblemWrapper::updateCallback(int mode, int n, const ColumnVector& x, double& fx, ColumnVector& gx, int& result, void* data)
{
    reinterpret_cast<UnconstrainedTimeIndexedProblemWrapper*>(data)->update(mode, n, x, fx, gx, result);
}

void UnconstrainedTimeIndexedProblemWrapper::updateCallbackFD(int n, const ColumnVector& x, double& fx, int& result, void* data)
{
    ColumnVector gx;
    reinterpret_cast<UnconstrainedTimeIndexedProblemWrapper*>(data)->update(NLPFunction, n, x, fx, gx, result);
}

void UnconstrainedTimeIndexedProblemWrapper::update(int mode, int n, const ColumnVector& x_opp, double& fx, ColumnVector& gx, int& result)
{
    if(n!=n_) throw_pretty("Invalid OPT++ state size, expecting "<<n_<<" got "<<n);

    Eigen::VectorXd x(problem_->N);
    Eigen::VectorXd x_prev = problem_->getInitialTrajectory()[0];
    Eigen::VectorXd x_prev_prev = x_prev;
    double T = (double)problem_->T;
    double ct = 1.0/problem_->tau/T;

    Eigen::VectorXd dx;

    problem_->Update(x_prev, 0);
    fx = problem_->getScalarTaskCost(0);

    for(int t=1; t<problem_->T; t++)
    {
        for(int i=0; i<problem_->N; i++) x(i) = x_opp((t-1)*problem_->N+i+1);

        problem_->Update(x, t);
        dx = x - x_prev;

        if (mode & NLPFunction)
        {
            fx += problem_->getScalarCost(t)*ct +
                    ct*dx.transpose()*problem_->W*dx;
            result = NLPFunction;
        }

        if (mode & NLPGradient)
        {
            Eigen::VectorXd J = problem_->getScalarJacobian(t)*ct
                    + 2.0*ct*problem_->W*dx;
            for(int i=0; i<problem_->N; i++) gx((t-1)*problem_->N+i+1) = J(i);
            if(t>1)
            {
                J=-2.0*ct*problem_->W*dx;
                for(int i=0; i<problem_->N; i++) gx((t-2)*problem_->N+i+1) += J(i);
            }
            result = NLPGradient;
        }
        x_prev_prev = x_prev;
        x_prev = x;
    }

    // Store cost
    int iter = solver_->getIter();
    if (iter == 1) hasBeenInitialized = true;
    if (!hasBeenInitialized) iter = 0;
    // HIGHLIGHT_NAMED("UTIPW::update", "iter: " << iter << " cost: " << fx)
    problem_->setCostEvolution(iter, fx);
}

void UnconstrainedTimeIndexedProblemWrapper::init(int n, ColumnVector& x)
{
    if(n!=n_) throw_pretty("Invalid OPT++ state size, expecting "<<n_<<" got "<<n);
    const std::vector<Eigen::VectorXd>& init = problem_->getInitialTrajectory();
    x.ReSize(n);
    for(int t=1; t<problem_->T; t++)
        for(int i=0; i<problem_->N; i++)
            x((t-1)*problem_->N+i+1) = init[t](i);
    hasBeenInitialized = false;
}

std::shared_ptr<FDNLF1WrapperUTIP> UnconstrainedTimeIndexedProblemWrapper::getFDNLF1()
{
    return std::shared_ptr<FDNLF1WrapperUTIP>(new FDNLF1WrapperUTIP(*this));
}

std::shared_ptr<NLF1WrapperUTIP> UnconstrainedTimeIndexedProblemWrapper::getNLF1()
{
    return std::shared_ptr<NLF1WrapperUTIP>(new NLF1WrapperUTIP(*this));
}

NLF1WrapperUTIP::NLF1WrapperUTIP(const UnconstrainedTimeIndexedProblemWrapper& parent) : parent_(parent),
    NLF1(parent.n_, UnconstrainedTimeIndexedProblemWrapper::updateCallback, nullptr, (void*)nullptr)
{
    vptr = reinterpret_cast<UnconstrainedTimeIndexedProblemWrapper*>(&parent_);
}

void NLF1WrapperUTIP::initFcn()
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

FDNLF1WrapperUTIP::FDNLF1WrapperUTIP(const UnconstrainedTimeIndexedProblemWrapper& parent) : parent_(parent),
    FDNLF1(parent.n_, UnconstrainedTimeIndexedProblemWrapper::updateCallbackFD, nullptr, (void*)nullptr)
{
    vptr = reinterpret_cast<UnconstrainedTimeIndexedProblemWrapper*>(&parent_);
}

void FDNLF1WrapperUTIP::initFcn()
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
