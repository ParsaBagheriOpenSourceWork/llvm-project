//
// Created by Parsa Bagheri on 11/23/21.
//

#ifndef MLIR_DIALECT_LUMINOUS_LUMINOUSDIALECT_H
#define MLIR_DIALECT_LUMINOUS_LUMINOUSDIALECT_H

#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/Dialect/Luminous/IR/LuminousTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//namespace mlir {
//namespace luminous {
//class AsyncTokenType
//    : public Type::TypeBase<AsyncTokenType, Type, TypeStorage> {
//public:
//  // Used for generic hooks in TypeBase.
//  using Base::Base;
//};
//} // namespace luminous
//} // namespace mlir

#include "mlir/Dialect/Luminous/IR/LuminousOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Luminous/IR/LuminousOps.h.inc"

#endif // MLIR_DIALECT_LUMINOUS_LUMINOUSDIALECT_H
