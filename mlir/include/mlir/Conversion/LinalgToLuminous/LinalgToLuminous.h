//===- LinalgToLuminous.h - Linalg to Luminous pass -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LINALGTOLUMINOUS_LINALGTOLUMINOUS_H
#define MLIR_CONVERSION_LINALGTOLUMINOUS_LINALGTOLUMINOUS_H

#include <memory>

namespace mlir {
template <typename T>
class OperationPass;
class ModuleOp;
std::unique_ptr<OperationPass<ModuleOp>>
createConvertLinalgToLuminousLaunchPass();
} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOLUMINOUS_LINALGTOLUMINOUS_H
