//===- LuminousTypes.h - Luminous Dialect Types -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types for the Luminous dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LUMINOUS_IR_LUMINOUSTYPES_H
#define MLIR_DIALECT_LUMINOUS_IR_LUMINOUSTYPES_H

#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// Luminous Dialect Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Luminous/IR/LuminousOpsTypes.h.inc"

#endif // MLIR_DIALECT_LUMINOUS_IR_LUMINOUSTYPES_H
