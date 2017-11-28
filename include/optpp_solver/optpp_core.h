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

#ifndef OPTPP_CORE_H
#define OPTPP_CORE_H

#include <memory>

#include <optpp_catkin/Opt.h>
#include <optpp_catkin/NLP.h>
#include <optpp_catkin/NLF.h>
#include <optpp_catkin/newmat.h>
#include <exotica/Problems/UnconstrainedEndPoseProblem.h>
#include <exotica/Problems/UnconstrainedTimeIndexedProblem.h>

using namespace OPTPP;
using namespace NEWMAT;

namespace exotica
{

class UnconstrainedEndPoseProblemWrapper;
class UnconstrainedTimeIndexedProblemWrapper;
class NLF1WrapperUEPP;
class FDNLF1WrapperUEPP;
class NLF1WrapperUTIP;
class FDNLF1WrapperUTIP;

class UnconstrainedEndPoseProblemWrapper
{
public:
    UnconstrainedEndPoseProblemWrapper(UnconstrainedEndPoseProblem_ptr problem);
    static void updateCallback(int mode, int n, const ColumnVector& x, double& fx, ColumnVector& gx, int& result, void* data);
    static void updateCallbackFD(int n, const ColumnVector& x, double& fx, int& result, void* data);

    void setSolver(std::shared_ptr<OPTPP::OptimizeClass> solver);

    void update(int mode, int n, const ColumnVector& x, double& fx, ColumnVector& gx, int& result);
    void init(int n, ColumnVector& x);

    std::shared_ptr<FDNLF1WrapperUEPP> getFDNLF1();
    std::shared_ptr<NLF1WrapperUEPP> getNLF1();


    UnconstrainedEndPoseProblem_ptr problem_;
    int n_;
    std::shared_ptr<OPTPP::OptimizeClass> solver_;

    bool hasBeenInitialized = false;
};

class NLF1WrapperUEPP : public virtual NLF1
{
public:
    NLF1WrapperUEPP(const UnconstrainedEndPoseProblemWrapper& parent);
    virtual void initFcn();
    void setSolver(std::shared_ptr<OPTPP::OptimizeClass> solver) { parent_.setSolver(solver); };
protected:
    UnconstrainedEndPoseProblemWrapper parent_;
};

class FDNLF1WrapperUEPP : public virtual FDNLF1
{
public:
    FDNLF1WrapperUEPP(const UnconstrainedEndPoseProblemWrapper& parent);
    virtual void initFcn();
    void setSolver(std::shared_ptr<OPTPP::OptimizeClass> solver) { parent_.setSolver(solver); };
protected:
    UnconstrainedEndPoseProblemWrapper parent_;
};





class UnconstrainedTimeIndexedProblemWrapper
{
public:
    UnconstrainedTimeIndexedProblemWrapper(UnconstrainedTimeIndexedProblem_ptr problem);
    static void updateCallback(int mode, int n, const ColumnVector& x, double& fx, ColumnVector& gx, int& result, void* data);
    static void updateCallbackFD(int n, const ColumnVector& x, double& fx, int& result, void* data);

    void setSolver(std::shared_ptr<OPTPP::OptimizeClass> solver);

    void update(int mode, int n, const ColumnVector& x, double& fx, ColumnVector& gx, int& result);
    void init(int n, ColumnVector& x);

    std::shared_ptr<FDNLF1WrapperUTIP> getFDNLF1();
    std::shared_ptr<NLF1WrapperUTIP> getNLF1();


    UnconstrainedTimeIndexedProblem_ptr problem_;
    int n_;
    std::shared_ptr<OPTPP::OptimizeClass> solver_;

    bool hasBeenInitialized = false;
};

class NLF1WrapperUTIP : public virtual NLF1
{
public:
    NLF1WrapperUTIP(const UnconstrainedTimeIndexedProblemWrapper& parent);
    virtual void initFcn();
    void setSolver(std::shared_ptr<OPTPP::OptimizeClass> solver) { parent_.setSolver(solver); };
protected:
    UnconstrainedTimeIndexedProblemWrapper parent_;
};

class FDNLF1WrapperUTIP : public virtual FDNLF1
{
public:
    FDNLF1WrapperUTIP(const UnconstrainedTimeIndexedProblemWrapper& parent);
    virtual void initFcn();
    void setSolver(std::shared_ptr<OPTPP::OptimizeClass> solver) { parent_.setSolver(solver); };
protected:
    UnconstrainedTimeIndexedProblemWrapper parent_;
};

}

#endif // OPTPP_CORE_H
