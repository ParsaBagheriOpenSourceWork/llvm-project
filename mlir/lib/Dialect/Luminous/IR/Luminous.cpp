//===- Luminous.cpp - MLIR Luminous Operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::luminous;

#include "mlir/Dialect/Luminous/IR/LuminousOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Luminous/IR/LuminousOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// LuminousDialect
//===----------------------------------------------------------------------===//

void LuminousDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Luminous/IR/LuminousOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Luminous/IR/LuminousOpsTypes.cpp.inc"
      >();
}

Type LuminousDialect::parseType(DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  // Handle 'async token' types.
  if (keyword == "async_token")
    return AsyncTokenType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown luminous type: " + keyword);
  return Type();
}

void LuminousDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<AsyncTokenType>([&](Type) { os << "async_token"; })
      .Default(
          [](Type) { llvm_unreachable("unexpected 'luminous' type kind"); });
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
            DispatchOp::getKernelAttrName()))
      return success();

    // Check that `dispatch` refers to a well-formed kernel module.
    StringAttr kernelModuleName = dispatchOp.getKernelModuleName();
    auto kernelModule = module.lookupSymbol<LuminousModuleOp>(kernelModuleName);
    if (!kernelModule)
      return dispatchOp.emitOpError()
             << "kernel module '" << kernelModuleName.getValue()
             << "' is undefined";

    // Check that `dispatch` refers to a well-formed kernel function.
    Operation *kernelFunc = module.lookupSymbol(dispatchOp.kernelAttr());
    auto kernelFunction =
        dyn_cast_or_null<luminous::LuminousFuncOp>(kernelFunc);
    if (!kernelFunction)
      return dispatchOp.emitOpError("kernel function '")
             << dispatchOp.kernel() << "' is undefined";

    unsigned actualNumArguments = dispatchOp.getNumKernelOperands();
    unsigned expectedNumArguments = kernelFunction.getNumArguments();
    if (expectedNumArguments != actualNumArguments)
      return dispatchOp.emitOpError("got ")
             << actualNumArguments << " kernel operands but expected "
             << expectedNumArguments;

    auto functionType = kernelFunction.getType();
    for (unsigned i = 0; i < expectedNumArguments; ++i) {
      if (dispatchOp.getKernelOperand(i).getType() !=
          functionType.getInput(i)) {
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

  // Ensure that this module has a valid terminator.
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

/// Parses a Luminous function.
///
/// <operation> ::= `luminous.func` symbol-ref-id `(` argument-list `)`
///                 function-attributes? region
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
    return parser.emitError(signatureLocation)
           << "luminous.func requires named arguments";

  if (!resultAttrs.empty() || !resultTypes.empty())
    return parser.emitError(signatureLocation)
           << "luminous.func does not return anything";

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

/// Prints a Luminous Func op.
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

/// Hook for FunctionLike verifier.
LogicalResult LuminousFuncOp::verifyType() {
  Type type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");

  if (getType().getNumResults() != 0)
    return emitOpError() << "expected void return type";

  return success();
}

/// Verifies the body of the function.
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
// ReturnOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(luminous::ReturnOp returnOp) {
  LuminousFuncOp function = returnOp->getParentOfType<LuminousFuncOp>();

  FunctionType funType = function.getType();

  if (funType.getNumResults() != returnOp.operands().size())
    return returnOp.emitOpError()
        .append("expected ", funType.getNumResults(), " result operands")
        .attachNote(function.getLoc())
        .append("return type declared here");

  for (auto pair : llvm::enumerate(
           llvm::zip(function.getType().getResults(), returnOp.operands()))) {
    Type type;
    Value operand;
    std::tie(type, operand) = pair.value();
    if (type != operand.getType())
      return returnOp.emitOpError() << "unexpected type `" << operand.getType()
                                    << "' for operand #" << pair.index();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DispatchOp
//===----------------------------------------------------------------------===//

void DispatchOp::build(OpBuilder &builder, OperationState &result,
                       LuminousFuncOp kernelFunc,
                       ValueRange asyncDependencies,
                       ValueRange kernelShape,
                       ValueRange kernelOperands) {
  result.addOperands(asyncDependencies);
  result.addOperands(kernelShape);
  result.addOperands(kernelOperands);
  result.addTypes({AsyncTokenType::get(result.getContext())});
  auto kernelModule = kernelFunc->getParentOfType<LuminousModuleOp>();
  auto kernelSymbol =
      SymbolRefAttr::get(kernelModule.getNameAttr(),
                         {SymbolRefAttr::get(kernelFunc.getNameAttr())});
  result.addAttribute(getKernelAttrName(), kernelSymbol);
  SmallVector<int32_t, 3> segmentSizes{
      static_cast<int32_t>(asyncDependencies.size()),
      static_cast<int32_t>(kernelShape.size()),
      static_cast<int32_t>(kernelOperands.size())};
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr(segmentSizes));
}

/// The number of operands passed to the kernel function.
unsigned DispatchOp::getNumKernelOperands() {
  return getNumOperands() - asyncDependencies().size() - shape().size();
}

/// The name of the kernel's containing module.
StringAttr DispatchOp::getKernelModuleName() {
  return kernel().getRootReference();
}

/// The name of the kernel.
StringAttr DispatchOp::getKernelName() { return kernel().getLeafReference(); }

/// The i-th operand passed to the kernel function.
Value DispatchOp::getKernelOperand(unsigned i) {
  return getOperand(asyncDependencies().size() + shape().size() + i);
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

  auto kernelAttr = op->getAttrOfType<SymbolRefAttr>(op.getKernelAttrName());
  if (!kernelAttr)
    return op.emitOpError("symbol reference attribute '" +
                          op.getKernelAttrName() + "' must be specified");

  return success();
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

static ParseResult parseAsyncResult(OpAsmParser &parser, Type &asyncTokenType) {
  auto loc = parser.getCurrentLocation();
  if (parser.getNumResults() == 0)
    return parser.emitError(loc, "needs to be named");
  asyncTokenType = parser.getBuilder().getType<AsyncTokenType>();
  return success();
}

static ParseResult parseAsyncDependencies(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::OperandType> &asyncDependencies) {
  return parser.parseOperandList(asyncDependencies,
                                 OpAsmParser::Delimiter::OptionalSquare);
}

static ParseResult
parseDispatchShape(OpAsmParser &parser,
                   SmallVectorImpl<OpAsmParser::OperandType> &args) {
  return parser.parseOperandList(args, OpAsmParser::Delimiter::LessGreater);
}

static void printDispatchShape(OpAsmPrinter &printer, Operation *op,
                               OperandRange shape) {
  if (shape.empty())
    return;
  printer << "<";
  llvm::interleaveComma(shape, printer);
  printer << ">";
}
static void printAsyncResult(OpAsmPrinter &printer, Operation *op,
                             Type asyncTokenType) {
  return;
}

static void printAsyncDependencies(OpAsmPrinter &printer, Operation *op,
                                   OperandRange asyncDependencies) {
  if (asyncDependencies.empty())
    return;
  printer << "[";
  llvm::interleaveComma(asyncDependencies, printer);
  printer << "]";
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Luminous/IR/LuminousOps.cpp.inc"