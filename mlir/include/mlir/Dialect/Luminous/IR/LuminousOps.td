//===-- LuminousOps.td - Luminous operation definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the Luminous dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LUMINOUS_OPS
#define LUMINOUS_OPS

include "mlir/Dialect/Async/IR/AsyncTypes.td"
include "mlir/Dialect/Luminous/IR/LuminousDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


//===----------------------------------------------------------------------===//
// Luminous op definitions
//===----------------------------------------------------------------------===//

class Luminous_Op<string mnemonic, list<Trait> traits = []>
    : Op<Luminous_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Luminous ModuleOp
//===----------------------------------------------------------------------===//

/// TODO: I think we eventually will need to drop the HasDefaultDLTIDataLayout
/// when we figure out our devices data layout
def Luminous_LuminousModuleOp : Luminous_Op<"module", [
  DataLayoutOpInterface, HasDefaultDLTIDataLayout, IsolatedFromAbove,
  SymbolTable, Symbol, SingleBlockImplicitTerminator<"ModuleEndOp">
]> {
  let summary = "A top level compilation unit containing code to be run on a "
                "luminous device.";
  let description = [{
    ModuleOp contains routines that are intended to be run on a luminous device.

    ```
    luminous.module @symbol_name {
      luminous.func{}
      ...
      luminous.module_end
    }
    ```
  }];
  let builders = [OpBuilder<(ins "StringRef" : $name)>];
  let regions = (region SizedRegion<1> : $body);
  let hasCustomAssemblyFormat = 1;
  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// Luminous ModuleEndOp
//===----------------------------------------------------------------------===//

def Luminous_ModuleEndOp
    : Luminous_Op<"module_end", [Terminator, HasParent<"LuminousModuleOp">]> {
  let summary = "A pseudo op that marks the end of a luminous.module.";
  let description = [{This op terminates the only block inside the only region
                          of a `luminous.module`.}];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Luminous FuncOp
//===----------------------------------------------------------------------===//

def Luminous_LuminousFuncOp : Luminous_Op<"func", [
  HasParent<"LuminousModuleOp">, AutomaticAllocationScope, FunctionOpInterface,
  IsolatedFromAbove, Symbol
]> {
  let summary = "Function executable on a Luminous device";

  let description = [{
    ```
        op :: = `luminous.func` symbol-ref-id `(` argument-list `)`
                memory-attribution
                function-attributes?
                region
    ```

    Example:
    ```mlir
      luminous.func @foo(%arg0: index) attributes{qux: "quux"} {
        ...
      }
    ```

    The generic form illustrates the concept
    ```mlir
      "luminous.func"(%arg: index) {sym_name: "foo", qux : "quux"}
        ({^bb0(% arg0: index):
          ...
        }): (index)->()
    ```
  }];

  let arguments = (ins TypeAttrOf<FunctionType>:$function_type);
  let regions = (region AnyRegion : $body);
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
                 CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    /// FIXME: We should drive this via the ODS `type` param.
    FunctionType getFunctionType() {
      return function_type();
    }

    /// Hooks for the input/output type enumeration in FunctionLike .
    unsigned getNumFuncArguments() { return getFunctionType().getNumInputs(); }
    unsigned getNumFuncResults() { return getFunctionType().getNumResults(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Hook for FunctionLike verifier.
    LogicalResult verifyType();

    /// Verifies the body of the function.
    LogicalResult verifyBody();
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Luminous ReturnOp
//===----------------------------------------------------------------------===//

def Luminous_ReturnOp : Luminous_Op<"return", [HasParent<"LuminousFuncOp">,
                                               NoSideEffect, Terminator]> {
  let summary = "Terminator for Luminous functions.";
  let description = [{
    A terminator operation for regions that appear in the body of `luminous.func`.
    `luminous.return` takes no operands.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Luminous DispatchOp
//===----------------------------------------------------------------------===//

def Luminous_DispatchOp : Luminous_Op<"dispatch", [AttrSizedOperandSegments]>,
                          Arguments<(ins Variadic<Async_TokenType>:$asyncDependencies,
                                       SymbolRefAttr:$function,
                                       Variadic<AnyType>:$operands)>,
                          Results<(outs Async_TokenType:$asyncToken)> {
  let summary = "Launches a Luminous async function";

  let description = [{
    Example:

    ```mlir
      module attributes{luminous.container_module} {

        // This module creates a separate compilation unit.
        luminous.module @kernels{
          luminous.func @kernel(%arg0: memref<128xf32>){
            ...
            luminous.return
          }
        }

        func @function(%arg0: memref<128xf32>) {
          %t0 = ... : !async.token
          %t1 = luminous.dispatch [%t0] @kernels::@kernel(%arg0)
          ...
          return
        }
      }
    ```
  }];

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<(ins "LuminousFuncOp": $luminousFunc,
                               "ValueRange": $asyncDependencies,
                               "ValueRange": $funcOperands)>];

  let extraClassDeclaration = [{
    /// The number of operands passed to the kernel function.
    unsigned getNumFuncOperands();

    /// The name of the Luminous function's containing module.
    StringAttr getFuncModuleName();

    /// The name of the function.
    StringAttr getFuncName();

    /// The i-th operand passed to the function.
    Value getFuncOperand(unsigned i);

    /// The name of the symbol reference attribute specifying the kernel to launch.
    static StringRef getFuncAttrName() { return "function"; }

    // This needs to quietly verify if attributes with names defined below are
    // present since it is run before the verifier of this op.
    friend LogicalResult LuminousDialect::verifyOperationAttribute(Operation *op,
                                                                   NamedAttribute attr);
  }];

  let hasVerifier = 1;
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $function
    custom<DispatchOpOperands>($operands, type($operands))
    attr-dict}];
}

//===----------------------------------------------------------------------===//
// Luminous LaunchOp
//===----------------------------------------------------------------------===//

def Luminous_LaunchOp : Luminous_Op<"launch", [AttrSizedOperandSegments]>,
                        Arguments<(ins Variadic<Index>:$shape,
                                         Variadic<Index>:$step)> {
  let summary = "luminous launch op";

  let description = [{
    Example:

    ```mlir
        module {
            func @function(%arg0: memref<128xf32>) {
            }
        }
    ```
  }];

  let regions = (region AnyRegion:$body);
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [OpBuilder<(ins "ValueRange": $shape,
                                 "ValueRange": $step)>];
}

//===----------------------------------------------------------------------===//
// Luminous YieldOp
//===----------------------------------------------------------------------===//

def Luminous_YieldOp : Luminous_Op<"yield", [HasParent<"LaunchOp">,
                                             NoSideEffect, Terminator]> {
  let summary = "Terminator for Luminous LaunchOp.";
  let description = [{
    A terminator operation for regions that appear in the body of `luminous.launch`.
    Takes no operands.
  }];

  let assemblyFormat = "attr-dict";
}
#endif // LUMINOUS_OPS