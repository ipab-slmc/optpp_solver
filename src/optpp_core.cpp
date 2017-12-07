/*
 *      Author: Vladimir Ivan
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
#include <optpp_catkin/Constraint.h>
#include <optpp_catkin/CompoundConstraint.h>
#include <optpp_catkin/BoundConstraint.h>
#include <optpp_catkin/LinearEquation.h>
#include <optpp_catkin/LinearInequality.h>
#include <optpp_catkin/NonLinearEquation.h>
#include <optpp_catkin/NonLinearInequality.h>
#include <optpp_catkin/NonLinearConstraint.h>
#include <Eigen/Dense>

namespace exotica
{

template<>
void UnconstrainedEndPoseProblemWrapper::update(int mode, int n, const ColumnVector& x_opp, double& fx, ColumnVector& gx, int& result)
{
    if (n != n_) throw_pretty("Invalid OPT++ state size, expecting " << n_ << " got " << n);
    Eigen::VectorXd x(n);
    for (int i = 0; i < n; i++) x(i) = x_opp(i + 1);
    problem_->Update(x);

    if (mode & NLPFunction)
    {
        fx = problem_->getScalarCost();
        result = NLPFunction;
    }

    if (mode & NLPGradient)
    {
        Eigen::VectorXd J = problem_->getScalarJacobian();
        for (int i = 0; i < n; i++) gx(i + 1) = J(i);
        result = NLPGradient;
    }
    storeCost(fx);
}

template<>
void UnconstrainedEndPoseProblemWrapper::init(int n, ColumnVector& x)
{
    if (n != n_) throw_pretty("Invalid OPT++ state size, expecting " << n_ << " got " << n);
    n=n_ = problem_->N;
    Eigen::VectorXd x0 = problem_->applyStartState();
    x.ReSize(n);
    for (int i = 0; i < n; i++) x(i + 1) = x0(i);
    hasBeenInitialized = false;
}

template<>
CompoundConstraint* UnconstrainedEndPoseProblemWrapper::createConstraints() {return nullptr;}


template<>
void BoundedEndPoseProblemWrapper::update(int mode, int n, const ColumnVector& x_opp, double& fx, ColumnVector& gx, int& result)
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
    storeCost(fx);
}

template<>
void BoundedEndPoseProblemWrapper::init(int n, ColumnVector& x)
{
    n=n_ = problem_->N;
    Eigen::VectorXd x0 = problem_->applyStartState();
    x.ReSize(n);
    for(int i=0; i<n; i++) x(i+1) = x0(i);
    hasBeenInitialized = false;
}

template<>
CompoundConstraint* BoundedEndPoseProblemWrapper::createConstraints()
{
    ColumnVector lower(n_);
    ColumnVector upper(n_);
    std::vector<double> bounds = problem_->getBounds();
    if(bounds.size()==n_*2)
    {
        for(int i=0; i<n_; i++)
        {
            lower(i+1) = bounds[i];
            upper(i+1) = bounds[i+n_];
        }
        Constraint bc = new BoundConstraint(n_, lower, upper);
        return new CompoundConstraint(bc);
    }
    else
    {
        throw_pretty("Invalid BC");
    }
}


template<>
void UnconstrainedTimeIndexedProblemWrapper::update(int mode, int n, const ColumnVector& x_opp, double& fx, ColumnVector& gx, int& result)
{
    if (n != n_) throw_pretty("Invalid OPT++ state size, expecting " << n_ << " got " << n);

    Eigen::VectorXd x(problem_->N);
    Eigen::VectorXd x_prev = problem_->getInitialTrajectory()[0];

    Eigen::VectorXd dx;

    problem_->Update(x_prev, 0);
    fx = problem_->getScalarTaskCost(0);

    for (int t = 1; t < problem_->getT(); t++)
    {
        for (int i = 0; i < problem_->N; i++) x(i) = x_opp((t - 1) * problem_->N + i + 1);

        problem_->Update(x, t);
        dx = x - x_prev;

        if (mode & NLPFunction)
        {
            fx += problem_->getScalarTaskCost(t) + problem_->getScalarTransitionCost(t);
            result = NLPFunction;
        }

        if (mode & NLPGradient)
        {
            Eigen::VectorXd J_control = problem_->getScalarTransitionJacobian(t);
            Eigen::VectorXd J = problem_->getScalarTaskJacobian(t) + J_control;
            for (int i = 0; i < problem_->N; i++) gx((t - 1) * problem_->N + i + 1) = J(i);
            if (t > 1)
            {
                for (int i = 0; i < problem_->N; i++) gx((t - 2) * problem_->N + i + 1) += -J_control(i);
            }
            result = NLPGradient;
        }
        x_prev = x;
    }
    storeCost(fx);
}

template<>
void UnconstrainedTimeIndexedProblemWrapper::init(int n, ColumnVector& x)
{
    if (n != n_) throw_pretty("Invalid OPT++ state size, expecting " << n_ << " got " << n);
    n=n_ = problem_->N*(problem_->getT()-1);
    const std::vector<Eigen::VectorXd>& init = problem_->getInitialTrajectory();
    x.ReSize(n);
    for (int t = 1; t < problem_->getT(); t++)
        for (int i = 0; i < problem_->N; i++)
            x((t - 1) * problem_->N + i + 1) = init[t](i);
    hasBeenInitialized = false;
}

template<>
CompoundConstraint* UnconstrainedTimeIndexedProblemWrapper::createConstraints() {return nullptr;}

template<>
void BoundedTimeIndexedProblemWrapper::update(int mode, int n, const ColumnVector& x_opp, double& fx, ColumnVector& gx, int& result)
{
    if(n!=n_) throw_pretty("Invalid OPT++ state size, expecting "<<n_<<" got "<<n);
    fx = 0.0;

    Eigen::VectorXd x(problem_->N);
    Eigen::VectorXd x_prev = problem_->getInitialTrajectory()[0];
    Eigen::VectorXd x_prev_prev = x_prev;
    double T = (double)problem_->getT();
    double ct = 1.0/problem_->getTau()/T;

    Eigen::VectorXd dx;

    for(int t=1; t<problem_->getT(); t++)
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
    storeCost(fx);
}

template<>
void BoundedTimeIndexedProblemWrapper::init(int n, ColumnVector& x)
{
    n=n_ = problem_->N*(problem_->getT()-1);
    const std::vector<Eigen::VectorXd>& init = problem_->getInitialTrajectory();
    x.ReSize(n);
    for (int t = 1; t < problem_->getT(); t++)
        for (int i = 0; i < problem_->N; i++)
            x((t - 1) * problem_->N + i + 1) = init[t](i);
    hasBeenInitialized = false;
}

template<>
CompoundConstraint* BoundedTimeIndexedProblemWrapper::createConstraints()
{
    int T = problem_->getT()-1;
    int n = problem_->N;
    ColumnVector lower(n*T);
    ColumnVector upper(n*T);
    std::vector<double> bounds = problem_->getBounds();
    if(bounds.size()==n*2)
    {
        for(int t=0;t<T;t++)
        {
            for(int i=0; i<n; i++)
            {
                lower(t*n+i+1) = bounds[i];
                upper(t*n+i+1) = bounds[i+n];
            }
        }
        Constraint bc = new BoundConstraint(n*T, lower, upper);
        return new CompoundConstraint(bc);
    }
    else
    {
        throw_pretty("Invalid BC");
    }
}
}
