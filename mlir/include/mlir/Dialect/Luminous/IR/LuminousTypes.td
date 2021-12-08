//===- LuminousTypes.td - Luminous dialect types -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Luminous dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LUMINOUS_IR_ASYNCTYPES
#define MLIR_DIALECT_LUMINOUS_IR_ASYNCTYPES

include "mlir/Dialect/Luminous/IR/LuminousDialect.td"

//===----------------------------------------------------------------------===//
// Luminous types
//===----------------------------------------------------------------------===//


class Luminous_Type<string name, string typeMnemonic> :
    TypeDef<Luminous_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Luminous_AsyncTokenType : Luminous_Type<"AsyncToken", "async_token"> {
  let summary = "async token type";
  let description = [{
    `luminous.async_token` is a type returned by luminous operations, and it
    becomes `available` when the operations that created it is completed.
  }];
}

#endif // MLIR_DIALECT_LUMINOUS_IR_ASYNCTYPES