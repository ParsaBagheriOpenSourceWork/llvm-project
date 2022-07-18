//===- LuminousToLLVM.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LUMINOUSTOLLVM_H
#define LLVM_LUMINOUSTOLLVM_H

#include <memory>

namespace mlir {
template <typename T>
class OperationPass;
class ModuleOp;
std::unique_ptr<OperationPass<ModuleOp>>
createConvertLuminousToLLVMPass();
} // namespace mlir


#endif // LLVM_LUMINOUSTOLLVM_H