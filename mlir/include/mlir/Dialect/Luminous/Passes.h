//===- Passes.h - Luminous pass entry points --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LUMINOUS_PASSES_H_
#define MLIR_DIALECT_LUMINOUS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace luminous {

class DispatchableBlocks;
class DispatchableBlock;
using DispatchBuilderFn = std::function<void(LaunchOp, DispatchableBlocks &)>;
void defaultDispatchBuilderFn(LaunchOp, DispatchableBlocks &);

} // namespace luminous

std::unique_ptr<OperationPass<FuncOp>>
createLuminousKernelOutliningPass(mlir::luminous::DispatchBuilderFn fn =
                                      mlir::luminous::defaultDispatchBuilderFn);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Luminous/Passes.h.inc"
} // namespace mlir
#endif // MLIR_DIALECT_LUMINOUS_PASSES_H_
