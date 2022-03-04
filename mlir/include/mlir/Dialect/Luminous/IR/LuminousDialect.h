//===- LuminousDialect.h - Luminous Dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Luminous dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LUMINOUS_LUMINOUSDIALECT_H
#define MLIR_DIALECT_LUMINOUS_LUMINOUSDIALECT_H

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/IR/AsyncTypes.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#include "mlir/Dialect/Luminous/IR/LuminousOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Luminous/IR/LuminousOps.h.inc"

namespace mlir {
namespace luminous {
constexpr char maxMemoryAttrName[] = "linalg-max-memory-footprint";
constexpr char launchAttrName[] = "luminous-launch";
} // namespace luminous
} // namespace mlir

#endif // MLIR_DIALECT_LUMINOUS_LUMINOUSDIALECT_H
