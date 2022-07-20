//===- Luminous.cpp - MLIR Luminous Operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"

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
  if (!attr.getValue().isa<UnitAttr>() ||
      attr.getName() != getContainerModuleAttrName())
    return success();

  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("expected '")
           << getContainerModuleAttrName() << "' attribute to be attached to '"
           << ModuleOp::getOperationName() << '\'';

  const auto walkResult = module.walk([&module](
                                          DispatchOp dispatchOp) -> WalkResult {
    // Check that `dispatch` refers to a well-formed capsule function.
    const auto funcAttr =
        dispatchOp->getAttrOfType<SymbolRefAttr>(dispatchOp.getFuncAttrName());
    if (!funcAttr)
      return dispatchOp.emitOpError("symbol reference attribute '" +
                                    dispatchOp.getFuncAttrName() +
                                    "' must be specified");

    // Check that `dispatch` refers to a well-formed luminous module.
    const StringAttr luminousModuleName = dispatchOp.getFuncModuleName();
    auto *luminousModule = module.lookupSymbol(luminousModuleName);

    if (!luminousModule ||
        (!isa<LuminousModuleOp>(luminousModule) &&
         !(isa<ModuleOp>(luminousModule) &&
           luminousModule->hasAttr(LuminousDialect::getModuleAttrName()))))
      return dispatchOp.emitOpError()
             << "luminous.module '" << luminousModuleName.getValue()
             << "' is undefined";

    Operation *capsuleFunc = module.lookupSymbol(funcAttr);
    if (!capsuleFunc ||
        (!isa<luminous::LuminousFuncOp>(capsuleFunc) &&
         !capsuleFunc->hasAttr(LuminousDialect::getFuncAttrName())))
      return dispatchOp.emitOpError("capsule function '")
             << dispatchOp.function() << "' is undefined";

    if (auto capsuleFunction =
            dyn_cast<luminous::LuminousFuncOp>(capsuleFunc)) {
      const unsigned actualNumArguments = dispatchOp.getNumFuncOperands();
      const unsigned expectedNumArguments = capsuleFunction.getNumArguments();
      if (expectedNumArguments != actualNumArguments)
        return dispatchOp.emitOpError("got ")
               << actualNumArguments
               << " capsule function operands but expected "
               << expectedNumArguments;

      const auto functionType = capsuleFunction.getFunctionType();
      for (unsigned i = 0; i < expectedNumArguments; ++i) {
        if (dispatchOp.getFuncOperand(i).getType() !=
            functionType.getInput(i)) {
          return dispatchOp.emitOpError("type of function argument ")
                 << i << " does not match";
        }
      }
    }
    // TODO: we have to cover else case, dispatch op could refer to
    // func::funcOp or llvm::funcOp, but it's subject to change that's why I
    // haven't checked it here
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

ParseResult LuminousModuleOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // If module attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the module body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, {}))
    return failure();

  // Ensure that this module has a valid terminator
  LuminousModuleOp::ensureTerminator(*body, parser.getBuilder(),
                                     result.location);
  return success();
}

void LuminousModuleOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getName());
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     {mlir::SymbolTable::getSymbolAttrName()});
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
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
  result.addRegion();
}

ParseResult LuminousFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the function name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  bool isVariadic;

  if (failed(function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, entryArgs, isVariadic, resultTypes,
          resultAttrs)))
    return failure();

  auto signatureLocation = parser.getCurrentLocation();

  // Checking if we have named arguments
  for (auto &arg : entryArgs) {
    if (arg.ssaName.name.empty())
      return parser.emitError(signatureLocation) << "requires named arguments";
  }

  if (!resultAttrs.empty() || !resultTypes.empty())
    return parser.emitError(signatureLocation) << "expected no return type";

  // Construct the function type. More types will be added to the region, but
  // not to the function type.
  Builder &builder = parser.getBuilder();
  SmallVector<Type> argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(LuminousFuncOp::getTypeAttrName(), TypeAttr::get(type));

  // Parse attributes.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();
  function_interface_impl::addArgAndResultAttrs(builder, result, entryArgs,
                                                resultAttrs);

  // Parse the region. If no argument names were provided, take all names
  // (including those of attributions) from the entry block.
  auto *body = result.addRegion();
  return parser.parseRegion(*body, entryArgs);
}

void LuminousFuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getName());

  const auto type = getFunctionType();
  function_interface_impl::printFunctionSignature(p, *this, type.getInputs(),
                                                  /*isVariadic=*/false,
                                                  type.getResults());

  function_interface_impl::printFunctionAttributes(
      p, *this, type.getNumInputs(), type.getNumResults(), {});
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult LuminousFuncOp::verifyType() {
  const auto type = getFunctionTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError() << "requires '" << getTypeAttrName()
                         << "' attribute of function type";

  if (getFunctionType().getNumResults() != 0)
    return emitOpError() << "expected no return type";

  return success();
}

LogicalResult LuminousFuncOp::verifyBody() {
  const unsigned numFuncArguments = getNumArguments();
  const unsigned numBlockArguments = front().getNumArguments();

  if (numBlockArguments != numFuncArguments)
    return emitOpError() << "expected " << numFuncArguments
                         << " arguments to body region";

  ArrayRef<Type> funcArgTypes = getFunctionType().getInputs();
  for (unsigned i = 0; i < numFuncArguments; ++i) {
    const auto blockArgType = front().getArgument(i).getType();
    if (funcArgTypes[i] != blockArgType)
      return emitOpError() << "expected body region argument #" << i
                           << " to be of type " << funcArgTypes[i] << ", got "
                           << blockArgType;
  }
  return success();
}

LogicalResult LuminousFuncOp::verify() {
  if (failed(verifyBody()) || failed(verifyType()))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// DispatchOp
//===----------------------------------------------------------------------===//

void DispatchOp::build(OpBuilder &builder, OperationState &result,
                       LuminousFuncOp function, ValueRange asyncDependencies,
                       ValueRange funcOperands) {
  result.addOperands(asyncDependencies);
  result.addOperands(funcOperands);
  result.addTypes({TokenType::get(result.getContext())});
  auto luminousModule = function->getParentOfType<LuminousModuleOp>();
  const auto capsuleSymbol =
      SymbolRefAttr::get(luminousModule.getNameAttr(),
                         {SymbolRefAttr::get(function.getNameAttr())});
  result.addAttribute(getFuncAttrName(), capsuleSymbol);
  const SmallVector<int32_t, 2> segmentSizes{
      static_cast<int32_t>(asyncDependencies.size()),
      static_cast<int32_t>(funcOperands.size())};
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr(segmentSizes));
}

/// The number of operands passed to the capsule function.
unsigned DispatchOp::getNumFuncOperands() { return funcOperands().size(); }

/// The name of the capsule function's containing module.
StringAttr DispatchOp::getFuncModuleName() {
  return function().getRootReference();
}

/// The name of the capsule function.
StringAttr DispatchOp::getFuncName() { return function().getLeafReference(); }

/// The i-th operand passed to the capsule function.
Value DispatchOp::getFuncOperand(unsigned i) {
  return getOperand(asyncDependencies().size() + i);
}

LogicalResult DispatchOp::verify() {
  const auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError("expected to belong to a module");

  if (!module->getAttrOfType<UnitAttr>(
          LuminousDialect::getContainerModuleAttrName()))
    return emitOpError("expected the closest surrounding module to have the '" +
                       LuminousDialect::getContainerModuleAttrName() +
                       "' attribute");

  return success();
}

static ParseResult parseAsyncDependencies(
    OpAsmParser &parser, Type &asyncTokenType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &asyncDependencies) {
  const auto loc = parser.getCurrentLocation();
  if (parser.getNumResults() == 0)
    return parser.emitError(loc, "must have a result");
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

static ParseResult parseDispatchOpOperands(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &argNames,
    SmallVectorImpl<Type> &argTypes) {
  SmallVector<OpAsmParser::Argument> args;
  if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return failure();
  for (auto &arg : args) {
    argNames.push_back(arg.ssaName);
    argTypes.push_back(arg.type);
  }
  return success();
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
                     ValueRange shape, ValueRange step) {
  result.addOperands(shape);
  result.addOperands(step);
  result.addAttribute(
      LaunchOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(shape.size()),
                                static_cast<int32_t>(step.size())}));
  result.addRegion();
}

ParseResult LaunchOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  // Parsing the operands
  auto parseKeywordOperands = [&](StringRef keyword)
      -> FailureOr<SmallVector<OpAsmParser::UnresolvedOperand>> {
    SmallVector<OpAsmParser::UnresolvedOperand> vector;
    if (parser.parseKeyword(keyword) ||
        parser.parseOperandList(vector, /*requiredOperandCount=*/-1,
                                OpAsmParser::Delimiter::Paren) ||
        parser.resolveOperands(vector, builder.getIndexType(), result.operands))
      return failure();
    return vector;
  };

  auto shape = parseKeywordOperands("shape");
  if (failed(shape))
    return failure();
  auto step = parseKeywordOperands("step");
  if (failed(step))
    return failure();

  // Now parse the body.
  auto region = std::make_unique<Region>();
  if (parser.parseRegion(*region, {}))
    return failure();
  result.addRegion(std::move(region));

  // Set `operand_segment_sizes` attribute.
  result.addAttribute(
      LaunchOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(shape.getValue().size()),
                                static_cast<int32_t>(step.getValue().size())}));

  // Parse attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void LaunchOp::print(OpAsmPrinter &p) {
  p << " shape (" << shape() << ")"
    << " step (" << step() << ") ";
  p.printRegion(body(), /*printEntryBlockArgs=*/true);
  p.printOptionalAttrDict(
      getOperation()->getAttrs(),
      /*elidedAttrs=*/LaunchOp::getOperandSegmentSizeAttr());
}

LogicalResult LaunchOp::verify() {
  if (shape().size() != step().size())
    return emitOpError("shape and step variables must be the same size.");
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Luminous/IR/LuminousOps.cpp.inc"
