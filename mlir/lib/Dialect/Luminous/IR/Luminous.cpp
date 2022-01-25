//===- Luminous.cpp - MLIR Luminous Operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::luminous;
using namespace mlir::async;

#include "mlir/Dialect/Luminous/IR/LuminousOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// LuminousDialect
//===----------------------------------------------------------------------===//

void LuminousDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Luminous/IR/LuminousOps.cpp.inc"
      >();
}

LogicalResult LuminousDialect::verifyOperationAttribute(Operation *op,
                                                        NamedAttribute attr) {
  if (!attr.second.isa<UnitAttr>() ||
      attr.first != getContainerModuleAttrName())
    return success();

  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("expected '")
           << getContainerModuleAttrName() << "' attribute to be attached to '"
           << ModuleOp::getOperationName() << '\'';

  auto walkResult = module.walk([&module](DispatchOp dispatchOp) -> WalkResult {
    // Ignore launches that are nested more or less deep than functions in the
    // module we are currently checking.
    if (!dispatchOp->getParentOp() ||
        dispatchOp->getParentOp()->getParentOp() != module)
      return success();

    // Ignore launch ops with missing attributes here. The errors will be
    // reported by the verifiers of those ops.
    if (!dispatchOp->getAttrOfType<SymbolRefAttr>(
            DispatchOp::getFuncAttrName()))
      return success();

    // Check that `dispatch` refers to a well-formed kernel module.
    StringAttr luminousModuleName = dispatchOp.getFuncModuleName();
    auto luminousModule =
        module.lookupSymbol<LuminousModuleOp>(luminousModuleName);
    if (!luminousModule)
      return dispatchOp.emitOpError()
             << "kernel module '" << luminousModuleName.getValue()
             << "' is undefined";

    // Check that `dispatch` refers to a well-formed kernel function.
    Operation *kernelFunc = module.lookupSymbol(dispatchOp.functionAttr());
    auto kernelFunction =
        dyn_cast_or_null<luminous::LuminousFuncOp>(kernelFunc);
    if (!kernelFunction)
      return dispatchOp.emitOpError("kernel function '")
             << dispatchOp.function() << "' is undefined";

    unsigned actualNumArguments = dispatchOp.getNumFuncOperands();
    unsigned expectedNumArguments = kernelFunction.getNumArguments();
    if (expectedNumArguments != actualNumArguments)
      return dispatchOp.emitOpError("got ")
             << actualNumArguments << " kernel operands but expected "
             << expectedNumArguments;

    auto functionType = kernelFunction.getType();
    for (unsigned i = 0; i < expectedNumArguments; ++i) {
      if (dispatchOp.getFuncOperand(i).getType() != functionType.getInput(i)) {
        return dispatchOp.emitOpError("type of function argument ")
               << i << " does not match";
      }
    }

    return success();
  });

  return walkResult.wasInterrupted() ? failure() : success();
}

//===----------------------------------------------------------------------===//
// LuminousModuleOp
//===----------------------------------------------------------------------===//

void LuminousModuleOp::build(OpBuilder &builder, OperationState &result,
                             StringRef name) {
  ensureTerminator(*result.addRegion(), builder, result.location);
  result.attributes.push_back(builder.getNamedAttr(
      ::mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
}

static ParseResult parseLuminousModuleOp(OpAsmParser &parser,
                                         OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // If module attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the module body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, None, None))
    return failure();

  // Ensure that this module has a valid terminator
  LuminousModuleOp::ensureTerminator(*body, parser.getBuilder(),
                                     result.location);
  return success();
}

static void print(OpAsmPrinter &p, LuminousModuleOp op) {
  p << ' ';
  p.printSymbolName(op.getName());
  p.printOptionalAttrDictWithKeyword(op->getAttrs(),
                                     {SymbolTable::getSymbolAttrName()});
  p.printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

//===----------------------------------------------------------------------===//
// LuminousFuncOp
//===----------------------------------------------------------------------===//

void LuminousFuncOp::build(OpBuilder &builder, OperationState &result,
                           StringRef name, FunctionType type,
                           ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  result.addAttributes(attrs);
  Region *body = result.addRegion();
  Block *entryBlock = new Block;
  entryBlock->addArguments(type.getInputs());
  body->getBlocks().push_back(entryBlock);
}

static ParseResult parseLuminousFuncOp(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 8> entryArgs;
  SmallVector<NamedAttrList, 1> argAttrs;
  SmallVector<NamedAttrList, 1> resultAttrs;
  SmallVector<Type, 8> argTypes;
  SmallVector<Type, 4> resultTypes;
  bool isVariadic;

  // Parse the function name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  auto signatureLocation = parser.getCurrentLocation();
  if (failed(function_like_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, entryArgs, argTypes, argAttrs,
          isVariadic, resultTypes, resultAttrs)))
    return failure();

  if (entryArgs.empty() && !argTypes.empty())
    return parser.emitError(signatureLocation) << "requires named arguments";

  if (!resultAttrs.empty() || !resultTypes.empty())
    return parser.emitError(signatureLocation) << "does not expect return type";

  // Construct the function type. More types will be added to the region, but
  // not to the function type.
  Builder &builder = parser.getBuilder();
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(LuminousFuncOp::getTypeAttrName(), TypeAttr::get(type));

  // Parse attributes.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();
  function_like_impl::addArgAndResultAttrs(builder, result, argAttrs,
                                           resultAttrs);

  // Parse the region. If no argument names were provided, take all names
  // (including those of attributions) from the entry block.
  auto *body = result.addRegion();
  return parser.parseRegion(*body, entryArgs, argTypes);
}

static void printLuminousFuncOp(OpAsmPrinter &p, LuminousFuncOp op) {
  p << ' ';
  p.printSymbolName(op.getName());

  FunctionType type = op.getType();
  function_like_impl::printFunctionSignature(
      p, op.getOperation(), type.getInputs(),
      /*isVariadic=*/false, type.getResults());

  function_like_impl::printFunctionAttributes(
      p, op.getOperation(), type.getNumInputs(), type.getNumResults(), {});
  p.printRegion(op.getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult LuminousFuncOp::verifyType() {
  Type type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");

  if (getType().getNumResults() != 0)
    return emitOpError() << "expected no return type";

  return success();
}

LogicalResult LuminousFuncOp::verifyBody() {
  unsigned numFuncArguments = getNumArguments();
  unsigned numBlockArguments = front().getNumArguments();
  if (numBlockArguments < numFuncArguments)
    return emitOpError() << "expected at least " << numFuncArguments
                         << " arguments to body region";

  ArrayRef<Type> funcArgTypes = getType().getInputs();
  for (unsigned i = 0; i < numFuncArguments; ++i) {
    Type blockArgType = front().getArgument(i).getType();
    if (funcArgTypes[i] != blockArgType)
      return emitOpError() << "expected body region argument #" << i
                           << " to be of type " << funcArgTypes[i] << ", got "
                           << blockArgType;
  }
  return success();
}

static LogicalResult verify(LuminousFuncOp op) {
  if (failed(op.verifyBody()) || failed(op.verifyType()))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// DispatchOp
//===----------------------------------------------------------------------===//

void DispatchOp::build(OpBuilder &builder, OperationState &result,
                       LuminousFuncOp function, ValueRange asyncDependencies,
                       ValueRange kernelOperands) {
  result.addOperands(asyncDependencies);
  result.addOperands(kernelOperands);
  result.addTypes({TokenType::get(result.getContext())});
  auto kernelModule = function->getParentOfType<LuminousModuleOp>();
  auto kernelSymbol = SymbolRefAttr::get(
      kernelModule.getNameAttr(), {SymbolRefAttr::get(function.getNameAttr())});
  result.addAttribute(getFuncAttrName(), kernelSymbol);
  SmallVector<int32_t, 3> segmentSizes{
      static_cast<int32_t>(asyncDependencies.size()),
      static_cast<int32_t>(kernelOperands.size())};
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr(segmentSizes));
}

/// The number of operands passed to the kernel function.
unsigned DispatchOp::getNumFuncOperands() {
  return getNumOperands() - asyncDependencies().size();
}

/// The name of the kernel's containing module.
StringAttr DispatchOp::getFuncModuleName() {
  return function().getRootReference();
}

/// The name of the kernel.
StringAttr DispatchOp::getFuncName() { return function().getLeafReference(); }

/// The i-th operand passed to the kernel function.
Value DispatchOp::getFuncOperand(unsigned i) {
  return getOperand(asyncDependencies().size() + i);
}

static LogicalResult verify(DispatchOp op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return op.emitOpError("expected to belong to a module");

  if (!module->getAttrOfType<UnitAttr>(
          LuminousDialect::getContainerModuleAttrName()))
    return op.emitOpError(
        "expected the closest surrounding module to have the '" +
        LuminousDialect::getContainerModuleAttrName() + "' attribute");

  auto kernelAttr = op->getAttrOfType<SymbolRefAttr>(op.getFuncAttrName());
  if (!kernelAttr)
    return op.emitOpError("symbol reference attribute '" +
                          op.getFuncAttrName() + "' must be specified");

  return success();
}

static ParseResult parseAsyncDependencies(
    OpAsmParser &parser, Type &asyncTokenType,
    SmallVectorImpl<OpAsmParser::OperandType> &asyncDependencies) {
  auto loc = parser.getCurrentLocation();
  if (parser.getNumResults() == 0)
    return parser.emitError(loc, "needs to be named");
  asyncTokenType = parser.getBuilder().getType<TokenType>();
  return parser.parseOperandList(asyncDependencies,
                                 OpAsmParser::Delimiter::OptionalSquare);
}

static void printAsyncDependencies(OpAsmPrinter &printer, Operation *op,
                                   Type asyncTokenType,
                                   OperandRange asyncDependencies) {
  if (asyncDependencies.empty())
    return;
  printer << "[";
  llvm::interleaveComma(asyncDependencies, printer);
  printer << "]";
}

static ParseResult
parseDispatchOpOperands(OpAsmParser &parser,
                        SmallVectorImpl<OpAsmParser::OperandType> &argNames,
                        SmallVectorImpl<Type> &argTypes) {
  SmallVector<NamedAttrList, 4> argAttrs;
  bool isVariadic = false;
  return function_like_impl::parseFunctionArgumentList(
      parser, /*allowAttributes=*/false,
      /*allowVariadic=*/false, argNames, argTypes, argAttrs, isVariadic);
}

static void printDispatchOpOperands(OpAsmPrinter &printer, Operation *,
                                    OperandRange operands, TypeRange types) {
  if (operands.empty())
    return;
  printer << "(";
  llvm::interleaveComma(llvm::zip(operands, types), printer,
                        [&](const auto &pair) {
                          printer.printOperand(std::get<0>(pair));
                          printer << ": ";
                          printer.printType(std::get<1>(pair));
                        });
  printer << ")";
}

//===----------------------------------------------------------------------===//
// LaunchOp
//===----------------------------------------------------------------------===//

void LaunchOp::build(OpBuilder &builder, OperationState &result,
                     ValueRange problemSize, ValueRange blockSize,
                     ValueRange operands) {
  result.addOperands(operands);
  result.addOperands(blockSize);
  Region *region = result.addRegion();
  Block *block = new Block();
  block->addArguments(blockSize);
  block->addArguments(operands);
  region->push_back(block);
  SmallVector<int32_t, 2> segmentSizes{
      static_cast<int32_t>(blockSize.size()),
      static_cast<int32_t>(operands.size())};
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr(segmentSizes));
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Luminous/IR/LuminousOps.cpp.inc"