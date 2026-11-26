// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

struct CusolverCholeskyOp : Op {
    Var* a, * factor;
    bool upper;
    CusolverCholeskyOp(Var* a, bool upper);

    const char* name() const override { return "cusolver_cholesky"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
