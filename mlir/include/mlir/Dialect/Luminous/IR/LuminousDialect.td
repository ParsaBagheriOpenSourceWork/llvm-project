//===- LuminousDialect.td ----------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Luminous dialect definition.
//
//===----------------------------------------------------------------------===//

#ifndef LUMINOUS_DIALECT_TD
#define LUMINOUS_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Luminous dialect definitions
//===----------------------------------------------------------------------===//

def Luminous_Dialect : Dialect {
  let name = "luminous";
  let cppNamespace = "::mlir::luminous";
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate the modules that contain
    /// kernel modules.
    static StringRef getContainerModuleAttrName() {
      return "luminous.container_module";
    }
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "luminous.kernel"; }
  }];
}

#endif // LUMINOUS_DIALECT_TD