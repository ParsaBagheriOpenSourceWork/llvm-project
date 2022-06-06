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

class DispatchBlocks;

namespace detail {
struct DispatchBlockImpl;
struct DispatchBlocksImpl;
} // namespace detail

/// A dispatch block is a basic block that will be put inside a luminous kernel
/// The only interface functionality that this class provides is pushing back an
/// op to the basic block. cloning and handling arguments to the kernel launch
/// is handled by the implementation
class DispatchBlock {
  friend struct detail::DispatchBlockImpl;
  detail::DispatchBlockImpl &impl;

public:
  DispatchBlock(detail::DispatchBlockImpl &theImpl) : impl(theImpl) {}
  /// Clones `op' and inserts it in the basic block; it keeps track of
  /// the op for performing replacement with dispatch call.
  void pushBack(Operation *op);
};

/// A collection of dispatch blocks. An object of this class provides an
/// interface to create new dispatch blocks. The blocks within an object of this
/// class will each be be outlined in a luminous kernel by the luminous kernel
/// outlining pass
class DispatchBlocks {
  detail::DispatchBlocksImpl &impl;

public:
  DispatchBlocks(detail::DispatchBlocksImpl &theImpl) : impl(theImpl) {}
  DispatchBlock addNewBlock(llvm::ArrayRef<DispatchBlock> dependencies = {},
                            const std::string &name = "");
};

/// A user defined function for deciding which ops within LaunchOp needs to be
/// dispatched
using DispatchBuilderFn = std::function<void(LaunchOp, DispatchBlocks &)>;

/// The default dispatch builder function dispatches all of the ops with a
/// certain attribute
void defaultDispatchBuilderFn(LaunchOp, DispatchBlocks &);

} // namespace luminous

std::unique_ptr<OperationPass<func::FuncOp>>
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
