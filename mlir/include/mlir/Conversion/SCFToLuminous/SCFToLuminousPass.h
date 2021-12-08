//===- SCFToLuminousPass.h - Converts loops to luminous kernels -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SCFTOLUMINOUS_SCFTOLUMINOUSPASS_H
#define MLIR_CONVERSION_SCFTOLUMINOUS_SCFTOLUMINOUSPASS_H

#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir {
class FuncOp;
template <typename T>
class OperationPass;
class Pass;

/// Creates a pass that converts scf.parallel operations into a
/// luminous.dispatch operation.
std::unique_ptr<Pass> createParallelLoopToGpuPass();

} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOLUMINOUS_SCFTOLUMINOUSPASS_H
