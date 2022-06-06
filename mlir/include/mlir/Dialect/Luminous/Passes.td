//===-- Passes.td - Luminous pass definition file ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LUMINOUS_PASSES
#define MLIR_DIALECT_LUMINOUS_PASSES

include "mlir/Pass/PassBase.td"

def LuminousKernelOutlining : Pass<"luminous-kernel-outlining", "func::FuncOp"> {
  let summary = "Outlines luminous.func and luminous.module for luminous"
                ".launch operations";
  let constructor = "mlir::createLuminousKernelOutliningPass()";
  let dependentDialects = [
    "luminous::LuminousDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "async::AsyncDialect"
  ];
}

#endif //MLIR_DIALECT_LUMINOUS_PASSES