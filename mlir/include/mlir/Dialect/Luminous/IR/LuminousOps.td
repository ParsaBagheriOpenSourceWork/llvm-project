//===-- OpenMPOps.td - OpenMP dialect operation definitions *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the OpenMP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LUMINOUS_OPS
#define LUMINOUS_OPS

include "mlir/Dialect/Luminous/IR/LuminousTypes.td"
include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/Dialect/Luminous/IR/LuminousDialect.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"


//===----------------------------------------------------------------------===//
// Luminous op definitions
//===----------------------------------------------------------------------===//

class Luminous_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Luminous_Dialect, mnemonic, traits>;

def Luminous_LuminousModuleOp : Luminous_Op<"module", [
  DataLayoutOpInterface, HasDefaultDLTIDataLayout, IsolatedFromAbove,
  SymbolTable, Symbol, SingleBlockImplicitTerminator<"ModuleEndOp">
]> {
  let summary = "A top level compilation unit containing code to be run on a "
                "luminous device.";
  let description = [{
    module contains code that is intended to be run on device.

    ```
    luminous.module @symbol_name {
      luminous.func{}
      ...
      luminous.module_end
    }
    ```
  }];
  let builders = [OpBuilder<(ins "StringRef" : $name)>];
  let parser = [{ return ::parseLuminousModuleOp(parser, result); }];
  let printer = [{ return ::print(p, *this); }];
  let regions = (region SizedRegion<1> : $body);

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;
}

def Luminous_ModuleEndOp
    : Luminous_Op<"module_end", [Terminator, HasParent<"LuminousModuleOp">]> {
  let summary = "A pseudo op that marks the end of a luminous.module.";
  let description = [{This op terminates the only block inside the only region
                          of a `gpu.module`.}];

  let assemblyFormat = "attr-dict";
}

def Luminous_DispatchOp : Luminous_Op<"dispatch", [AttrSizedOperandSegments]>,
    Arguments<(ins Variadic<Luminous_AsyncTokenType>:$asyncDependencies,
                 SymbolRefAttr:$kernel,
                 Variadic<Index>:$shape,
                 Variadic<AnyType>:$operands)>,
    Results<(outs Luminous_AsyncTokenType:$asyncToken)> {
  let summary = "Launches a function as a Luminous kernel";

  let description = [{
    Example:

    ```mlir
    module attributes{luminous.container_module} {

        // This module creates a separate compilation unit.
        luminous.module @kernels{
            luminous.func @kernel_0() {
              luminous.return
            }

            luminous.func @kernel_1(%arg0: memref<128xf32>){
              ...
              luminous.return
            }
        }

        func @function(%arg0: memref<128xf32>) {
          %c_128 = constant 128: index
          %t0 = luminous.dispatch @kernels::@kernel_0 <%c_128> ()
          %t1 = luminous.dispatch
                [% t0] // Async dependency tokens
                @kernels::@kernel_1 // Kernel function symbol
                <%c_128> // Kernel shape
                (%arg0) // Kernel operands
                : !luminous.async_token
          luminous.wait [%t1]
          return
        }
    }
    ```
  }];

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<(ins "LuminousFuncOp": $kernelFunc,
                                 "ValueRange": $asyncDependencies,
                                 "ValueRange": $kernelShape,
                                 "ValueRange": $kernelOperands)>];

  let extraClassDeclaration = [{
    /// The number of operands passed to the kernel function.
    unsigned getNumKernelOperands();

    /// The name of the kernel's containing module.
    StringAttr getKernelModuleName();

    /// The name of the kernel.
    StringAttr getKernelName();

    /// The i-th operand passed to the kernel function.
    Value getKernelOperand(unsigned i);

    /// The name of the symbol reference attribute specifying the kernel to launch.
    static StringRef getKernelAttrName() { return "kernel"; }

    // This needs to quietly verify if attributes with names defined below are
    // present since it is run before the verifier of this op.
    friend LogicalResult LuminousDialect::verifyOperationAttribute(Operation *op,
                                                                   NamedAttribute attr);
  }];

  let verifier = [{ return ::verify(*this); }];
  let assemblyFormat = [{
      custom<AsyncResult>(type($asyncToken))
      custom<AsyncDependencies>($asyncDependencies)
      $kernel
      custom<DispatchShape>($shape)
      custom<DispatchOpOperands>($operands, type($operands))
      attr-dict}];
}

def Luminous_LuminousFuncOp : Luminous_Op<"func", [
  HasParent<"LuminousModuleOp">, AutomaticAllocationScope, FunctionLike,
  IsolatedFromAbove, Symbol
]> {
  let summary = "Function executable on a Luminous device";

  let description = [{
    ``` op :: = `luminous.func` symbol-ref-id `(` argument-list `)`
                 (`->` function-result-list)?
                  memory-attribution `kernel`?
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
      "luminous.func"(% arg: index) {sym_name: "foo", kernel, qux : "quux"}
        ({^bb0(% arg0: index):
          ...
    }): (index)->()
    ```
  }];

  let regions = (region AnyRegion : $body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
                 CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    // FunctionLike trait needs access to the functions below.
    friend class OpTrait::FunctionLike<LuminousFuncOp>;

    /// Hooks for the input/output type enumeration in FunctionLike .
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Hook for FunctionLike verifier.
    LogicalResult verifyType();

    /// Verifies the body of the function.
    LogicalResult verifyBody();
  }];

  let verifier = [{ return ::verify(*this); }];
  let printer = [{ printLuminousFuncOp(p, *this); }];
  let parser = [{ return parseLuminousFuncOp(parser, result); }];
}

def Luminous_WaitOp : Luminous_Op<"wait"> {
  let summary = "Wait for async ops to complete.";
  let description = [{
    This op synchronizes the host or the device with a list of dependent
  }];

  let arguments = (ins Variadic<Luminous_AsyncTokenType> : $asyncDependencies);
  let results = (outs);

  let assemblyFormat =
      [{custom<AsyncDependencies>($asyncDependencies)
          attr-dict}];
}

def Luminous_ReturnOp : Luminous_Op<"return", [HasParent<"LuminousFuncOp">, NoSideEffect,
                                     Terminator]>,
                   Arguments<(ins Variadic<AnyType>:$operands)>, Results<(outs)> {
  let summary = "Terminator for Luminous functions.";
  let description = [{
    A terminator operation for regions that appear in the body of  `luminous.func`
    functions. The operands to the `luminous.return` are the result values returned
    by an invocation of the `luminous.func`.
  }];

  let builders = [OpBuilder<(ins), [{ // empty}]>];

    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
    let verifier = [{ return ::verify(*this); }];
}

#endif // LUMINOUS_OPS