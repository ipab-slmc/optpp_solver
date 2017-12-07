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

#ifndef OPTPP_CORE_H
#define OPTPP_CORE_H

#include <memory>

#include <exotica/Problems/UnconstrainedEndPoseProblem.h>
#include <exotica/Problems/UnconstrainedTimeIndexedProblem.h>
#include <optpp_catkin/NLF.h>
#include <optpp_catkin/NLP.h>
#include <optpp_catkin/Opt.h>
#include <optpp_catkin/newmat.h>
#include <exotica/Problems/BoundedEndPoseProblem.h>
#include <exotica/Problems/BoundedTimeIndexedProblem.h>
#include <exotica/Problems/EndPoseProblem.h>
#include <exotica/Problems/TimeIndexedProblem.h>

using namespace OPTPP;
using namespace NEWMAT;

namespace exotica
{

template<typename ProblemType> class NLF1Wrapper;
template<typename ProblemType> class NLF1WrapperFD;
template<typename ProblemType> class ProblemWrapper;

template<typename ProblemType>
class ProblemWrapper : public std::enable_shared_from_this<ProblemWrapper<ProblemType>>
{
public:
    ProblemWrapper(std::shared_ptr<ProblemType> problem):
        problem_(problem), n_(0)
    {
        ColumnVector dummy;
        init(0, dummy);
        constrains_ = createConstraints();
    }

    static void updateCallback(int mode, int n, const ColumnVector& x, double& fx, ColumnVector& gx, int& result, void* data)
    {
        reinterpret_cast<ProblemWrapper<ProblemType>*>(data)->update(mode, n, x, fx, gx, result);
        if(Server::isRos() && !ros::ok()) throw_pretty("OPTPPP solver interrupted!");
    }

    static void updateCallbackFD(int n, const ColumnVector& x, double& fx, int& result, void* data)
    {
        ColumnVector gx;
        reinterpret_cast<ProblemWrapper<ProblemType>*>(data)->update(NLPFunction, n, x, fx, gx, result);
        if(Server::isRos() && !ros::ok()) throw_pretty("OPTPPP solver interrupted!");
    }

    void setSolver(std::shared_ptr<OPTPP::OptimizeClass> solver);

    void update(int mode, int n, const ColumnVector& x, double& fx, ColumnVector& gx, int& result);
    void init(int n, ColumnVector& x);
    CompoundConstraint* createConstraints();

    std::shared_ptr<NLF1WrapperFD<ProblemType>> getFDNLF1()
    {
        return std::shared_ptr<NLF1WrapperFD<ProblemType>>(new NLF1WrapperFD<ProblemType>(this->shared_from_this()));
    }

    std::shared_ptr<NLF1Wrapper<ProblemType>> getNLF1()
    {
        return std::shared_ptr<NLF1Wrapper<ProblemType>>(new NLF1Wrapper<ProblemType>(this->shared_from_this()));
    }


    std::shared_ptr<ProblemType> problem_;
    int n_;
    CompoundConstraint* constrains_;
};

template<typename ProblemType>
class NLF1Wrapper : public virtual NLF1
{
public:
    NLF1Wrapper(std::shared_ptr<ProblemWrapper<ProblemType>> parent): parent_(parent),
        NLF1(parent.n_, ProblemWrapper<ProblemType>::updateCallback, nullptr, parent.constrains_, (void*)nullptr)
    {
        vptr = reinterpret_cast<ProblemWrapper<ProblemType>*>(parent_.get());
    }

    virtual void initFcn()
    {
        if (init_flag == false)
        {
            parent_->init(dim, mem_xc);
            init_flag = true;
        }
        else
        {
          parent_->init(dim, mem_xc);
        }
    }
protected:
    std::shared_ptr<ProblemWrapper<ProblemType>> parent_;
};

template<typename ProblemType>
class NLF1WrapperFD : public virtual FDNLF1
{
public:
    NLF1WrapperFD(std::shared_ptr<ProblemWrapper<ProblemType>> parent): parent_(parent),
        FDNLF1(parent.n_, ProblemWrapper<ProblemType>::updateCallbackFD, nullptr, parent.constrains_, (void*)nullptr)
    {
        vptr = reinterpret_cast<ProblemWrapper<ProblemType>*>(parent_.get());
    }

    virtual void initFcn()
    {
        if (init_flag == false)
        {
            parent_->init(dim, mem_xc);
            init_flag = true;
        }
        else
        {
          parent_->init(dim, mem_xc);
        }
    }
protected:
    std::shared_ptr<ProblemWrapper<ProblemType>> parent_;
};

typedef ProblemWrapper<UnconstrainedEndPoseProblem> UnconstrainedEndPoseProblemWrapper;
typedef ProblemWrapper<UnconstrainedTimeIndexedProblem> UnconstrainedTimeIndexedProblemWrapper;
typedef ProblemWrapper<BoundedEndPoseProblem> BoundedEndPoseProblemWrapper;
typedef ProblemWrapper<BoundedTimeIndexedProblem> BoundedTimeIndexedProblemWrapper;
//typedef ProblemWrapper<EndPoseProblem> EndPoseProblemWrapper;
//typedef ProblemWrapper<TimeIndexedProblem> TimeIndexedProblemWrapper;
}

#endif  // OPTPP_CORE_H
