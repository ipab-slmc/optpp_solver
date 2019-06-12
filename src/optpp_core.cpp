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

#include <optpp_solver/optpp_core.h>
#include <Eigen/Dense>

namespace exotica
{
UnconstrainedEndPoseProblemWrapper::UnconstrainedEndPoseProblemWrapper(UnconstrainedEndPoseProblemPtr problem) : problem_(problem), n_(problem_->N)
{
    if (problem_->GetNominalPose().rows() > 0) ThrowPretty("OPT++ solvers don't support null-space optimization! " << problem_->GetNominalPose().rows());
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
    if (n != n_) ThrowPretty("Invalid OPT++ state size, expecting " << n_ << " got " << n);
    Eigen::VectorXd x(n);
    for (int i = 0; i < n; i++) x(i) = x_opp(i + 1);
    problem_->Update(x);

    if (mode & NLPFunction)
    {
        fx = problem_->GetScalarCost();
        result = NLPFunction;
    }

    if (mode & NLPGradient)
    {
        const auto J = problem_->GetScalarJacobian();
        for (int i = 0; i < n; i++) gx(i + 1) = J(i);
        result = NLPGradient;
    }

    // Store cost
    int iter = solver_->getIter();
    if (iter == 1) hasBeenInitialized = true;
    if (!hasBeenInitialized) iter = 0;
    if (mode & NLPFunction)
    {
        // HIGHLIGHT_NAMED("UEPPW::update", "mode: " << mode << " iter: " << iter << " cost: " << fx << " (internal solver iter=" << solver_->getIter() << ")");
        problem_->SetCostEvolution(iter, fx);
    }
}

void UnconstrainedEndPoseProblemWrapper::init(int n, ColumnVector& x)
{
    if (n != n_) ThrowPretty("Invalid OPT++ state size, expecting " << n_ << " got " << n);
    Eigen::VectorXd x0 = problem_->ApplyStartState();
    x.ReSize(n);
    for (int i = 0; i < n; i++) x(i + 1) = x0(i);
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

UnconstrainedTimeIndexedProblemWrapper::UnconstrainedTimeIndexedProblemWrapper(UnconstrainedTimeIndexedProblemPtr problem) : problem_(problem), n_(problem_->N * (problem_->GetT() - 1))
{
}

UnconstrainedTimeIndexedProblemWrapper::UnconstrainedTimeIndexedProblemWrapper(UnconstrainedTimeIndexedProblemPtr problem, bool isLBFGS_in) : problem_(problem), n_(problem_->N * (problem_->GetT() - 1)), isLBFGS(isLBFGS_in)
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
    if (n != n_) ThrowPretty("Invalid OPT++ state size, expecting " << n_ << " got " << n);

    Eigen::VectorXd x = problem_->GetInitialTrajectory()[0];
    problem_->Update(x, 0);

    // Do not store initial state task cost as we are not optimising the initial configuration, cf.
    // https://github.com/ipab-slmc/exotica/issues/297
    // if (mode & NLPFunction) fx = problem_->GetScalarTaskCost(0);
    fx = 0;

    for (int t = 1; t < problem_->GetT(); t++)
    {
        for (int i = 0; i < problem_->N; i++) x(i) = x_opp((t - 1) * problem_->N + i + 1);
        problem_->Update(x, t);

        if (mode & NLPFunction)
        {
            fx += problem_->GetScalarTaskCost(t) + problem_->GetScalarTransitionCost(t);
            result = NLPFunction;
        }

        if (mode & NLPGradient)
        {
            Eigen::VectorXd J_control = problem_->GetScalarTransitionJacobian(t);
            Eigen::VectorXd J = problem_->GetScalarTaskJacobian(t) + J_control;
            for (int i = 0; i < problem_->N; i++) gx((t - 1) * problem_->N + i + 1) = J(i);
            if (t > 1)
            {
                for (int i = 0; i < problem_->N; i++) gx((t - 2) * problem_->N + i + 1) += -J_control(i);
            }
            result = NLPGradient;
        }
    }

    // Store cost
    int iter = solver_->getIter();
    if (!hasBeenInitialized)
    {
        iter = 0;
        hasBeenInitialized = true;
        problem_->SetCostEvolution(iter, fx);
        return;
    }

    // std::cout << "Iteration: " << iter << " - Mode: " << mode << ": " << fx << std::endl;
    if (mode & NLPFunction)
    {
        if (!isLBFGS && iter > 0)
        {
            // HIGHLIGHT_NAMED("UTIPW::update", "mode: " << mode << " iter: " << iter << " cost: " << fx << " (internal solver iter=" << solver_->getIter() << ")");
            problem_->SetCostEvolution(iter, fx);
        }
        else
        {
            problem_->SetCostEvolution(iter + 1, fx);
        }
    }
}

void UnconstrainedTimeIndexedProblemWrapper::init(int n, ColumnVector& x)
{
    if (n != n_) ThrowPretty("Invalid OPT++ state size, expecting " << n_ << " got " << n);
    const std::vector<Eigen::VectorXd>& init = problem_->GetInitialTrajectory();
    x.ReSize(n);
    for (int t = 1; t < problem_->GetT(); t++)
        for (int i = 0; i < problem_->N; i++)
            x((t - 1) * problem_->N + i + 1) = init[t](i);
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
}  // namespace exotica
