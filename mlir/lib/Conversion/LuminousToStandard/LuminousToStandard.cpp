//
// Created by parsab on 6/27/22.
//

#include "mlir/Conversion/LuminousToStandard/LuminousToStandard.h"
#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace luminous;

namespace {

constexpr char LuminousRuntimeDispatchFn[] = "__luminous_runtime_dispatch";

struct LuminousFuncToFunc : public ConvertOpToLLVMPattern<LuminousFuncOp> {
  using ConvertOpToLLVMPattern<LuminousFuncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LuminousFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template <typename Operands>
static LLVM::LLVMStructType
createArgStruct(PatternRewriter &rewriter, Operands &&operands,
                std::vector<Type> &paramTypes, std::vector<Value> &actuals,
                TypeConverter *typeConverter, StringRef name) {
  for (auto operand_tuple : operands) {
    auto operand = std::get<0>(operand_tuple);
    if (operand.getType().template isa<MemRefType>()) {
      llvm::SmallVector<Value, 5> memrefValues;
      auto operandAdaptor = std::get<1>(operand_tuple);
      MemRefDescriptor::unpack(
          rewriter, operand.getLoc(), operandAdaptor,
          operand.getType().template dyn_cast<MemRefType>(), memrefValues);
      for (auto m : memrefValues) {
        paramTypes.push_back(m.getType());
        actuals.push_back(m);
      }
    } else {
      auto llvmType = typeConverter->convertType(operand.getType());
      paramTypes.push_back(llvmType);
      actuals.push_back(operand);
    }
  }
  return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), paramTypes,
                                          /* packed */ true);
}

LogicalResult
LuminousFuncToFunc::matchAndRewrite(LuminousFuncOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {

  TypeConverter::SignatureConversion result(op.getNumArguments());
  auto llvmType = getTypeConverter()->convertFunctionSignature(
      op.getFunctionType(), false, result);
  if (!llvmType)
    return failure();

  auto module = op->getParentOfType<ModuleOp>();
  LLVM::LLVMFuncOp newFuncOp = LLVM::lookupOrCreateFn(
      module, op.getName(), llvmType.cast<LLVM::LLVMFunctionType>().getParams(),
      llvmType.cast<LLVM::LLVMFunctionType>().getReturnType());

  {
    OpBuilder::InsertionGuard innerGuard(rewriter);
    rewriter.setInsertionPointToEnd(&op.getBody().back());
    assert(llvm::hasSingleElement(op.getRegion()) &&
           "expected luminous.func to have one block");
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
        op.getBody().back().getTerminator(), ValueRange());
  }
  rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                         &result)))
    return failure();

  rewriter.eraseOp(op);
  return success();
}

struct LuminousModuleToStandard : public OpRewritePattern<LuminousModuleOp> {
  using OpRewritePattern<LuminousModuleOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LuminousModuleOp op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult
LuminousModuleToStandard::matchAndRewrite(LuminousModuleOp op,
                                          PatternRewriter &rewriter) const {
  auto moduleOp = rewriter.replaceOpWithNewOp<ModuleOp>(op, op.getName());
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());
  rewriter.eraseOp(op.getBody()->getTerminator());
  for (auto &bodyOp : op.body().getOps()) {
    if (!isa<luminous::ModuleEndOp>(bodyOp)) {
      rewriter.clone(bodyOp);
    }
  }
  moduleOp->setAttr("luminous.module", rewriter.getUnitAttr());
  return success();
}

struct LuminousDispatchToStandard : public ConvertOpToLLVMPattern<DispatchOp> {
  using ConvertOpToLLVMPattern<DispatchOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(DispatchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

static Value packArgsStruct(PatternRewriter &rewriter, Location loc,
                            LLVM::LLVMStructType paramStruct,
                            std::vector<Type> &paramTypes,
                            std::vector<Value> &actuals) {
  auto llvmInt32Type = IntegerType::get(rewriter.getContext(), 32);
  auto zero = rewriter.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                                rewriter.getI32IntegerAttr(0));
  auto one = rewriter.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                               rewriter.getI32IntegerAttr(1));
  auto structPtr = rewriter.create<LLVM::AllocaOp>(
      loc, LLVM::LLVMPointerType::get(paramStruct), one, 0);

  for (auto it : llvm::enumerate(actuals)) {
    auto i = it.index();
    auto v = it.value();
    auto index = rewriter.create<LLVM::ConstantOp>(
        loc, llvmInt32Type, rewriter.getI32IntegerAttr(i));
    auto fieldPtr = rewriter.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(paramTypes[i]), structPtr,
        ArrayRef<Value>{zero, index.getResult()});
    rewriter.create<LLVM::StoreOp>(loc, v, fieldPtr);
  }

  return structPtr.getRes();
}

LogicalResult LuminousDispatchToStandard::matchAndRewrite(
    DispatchOp op, OpAdaptor operands,
    ConversionPatternRewriter &rewriter) const {

  auto module = op->getParentOfType<ModuleOp>();
  OpBuilder::InsertionGuard g(rewriter);
  auto loc = op.getLoc();
  std::vector<Type> paramTypes;
  std::vector<Value> actuals;
  auto paramStruct = createArgStruct(
      rewriter, llvm::zip(op.getOperands(), operands.getOperands()), paramTypes,
      actuals, typeConverter, op.getFuncName().getValue());

  auto structPtr =
      packArgsStruct(rewriter, loc, paramStruct, paramTypes, actuals);

  /// TODO : temporary here ---------------
  auto ptrType = LLVM::LLVMPointerType::get(IntegerType::get(getContext(), 8));
  auto wrapperName = std::string(op.getFuncName().getValue()) + "_wrapper";
  auto capsule = LLVM::lookupOrCreateFn(module, wrapperName, {ptrType},
                                        LLVM::LLVMVoidType::get(getContext()));
  auto fnPtr = rewriter.create<LLVM::AddressOfOp>(op.getLoc(), capsule);
  auto fnPtrToVoidPtr = rewriter.create<LLVM::BitcastOp>(
      op.getLoc(), ptrType, ValueRange(fnPtr.getRes()));
  ///--------------------------------------

  auto runtime_fn = LLVM::lookupOrCreateFn(module, LuminousRuntimeDispatchFn,
                                           {ptrType, ptrType}, ptrType);

  auto paramsPtr = rewriter.create<LLVM::BitcastOp>(op.getLoc(), ptrType,
                                                    ValueRange(structPtr));
  rewriter.create<LLVM::CallOp>(
      op.getLoc(), runtime_fn,
      ValueRange({fnPtrToVoidPtr.getRes(), paramsPtr.getRes()}));

  /// TODO : temporary here ---------------

  for (auto user : op->getUsers())
    rewriter.eraseOp(user);

  ///--------------------------------------

  // Creating a wrapper function for unpacking arguments and calling kernel

  auto luminousModuleOp =
      module.lookupSymbol(op.getFuncModuleName().getValue());
  assert(luminousModuleOp && "luminous module not found");

  auto luminousModule = dyn_cast<ModuleOp>(luminousModuleOp);
  //  assert(luminousModule && "must be of luminous module type");

  rewriter.setInsertionPointToEnd(luminousModule.getBody(0));
  auto llvmWrapperFn = rewriter.create<LLVM::LLVMFuncOp>(
      loc, wrapperName,
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(getContext()),
                                  {ptrType}));

  auto llvmFnUnpacked = LLVM::lookupOrCreateFn(
      luminousModule, op.getFuncName().getValue(), paramStruct.getBody(),
      LLVM::LLVMVoidType::get(getContext()));
  Block *body = new Block;
  body->addArgument(ptrType, loc);
  {
    OpBuilder::InsertionGuard innerGuard(rewriter);
    rewriter.setInsertionPointToEnd(body);

    // Casting void ptr to a pointer to our struct type
    auto argBitCast = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(paramStruct),
        ValueRange(body->getArgument(0)));
    auto llvmInt32Type = IntegerType::get(rewriter.getContext(), 32);
    auto zero = rewriter.create<LLVM::ConstantOp>(
        loc, llvmInt32Type, rewriter.getI32IntegerAttr(0));

    // Unpacking arguments
    std::vector<Value> args;
    for (auto tuple : llvm::enumerate(paramStruct.getBody())) {
      auto t = tuple.value();
      int i = tuple.index();
      auto index = rewriter.create<LLVM::ConstantOp>(
          loc, llvmInt32Type, rewriter.getI32IntegerAttr(i));
      auto elmPtr = rewriter.create<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(t), argBitCast.getRes(),
          ValueRange({zero.getRes(), index.getRes()}));
      auto loadOp = rewriter.create<LLVM::LoadOp>(loc, elmPtr.getRes());
      args.push_back(loadOp.getRes());
    }

    // Passing arguments to the actual kernel
    rewriter.create<LLVM::CallOp>(loc, llvmFnUnpacked, args);
    rewriter.create<LLVM::ReturnOp>(loc, ValueRange());
  }
  llvmWrapperFn.getBody().push_back(body);

  rewriter.eraseOp(op);
  return success();
}

static LogicalResult applyPatterns(ModuleOp module) {
  ConversionTarget target(*module.getContext());
  target.addLegalDialect<LuminousDialect, linalg::LinalgDialect,
                         async::AsyncDialect>();
  target.addLegalDialect<BuiltinDialect, func::FuncDialect>();
  target.addIllegalOp<LuminousFuncOp, LuminousModuleOp, DispatchOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  RewritePatternSet patterns(module.getContext());
  LLVMTypeConverter tc(module.getContext());
  patterns.add<LuminousDispatchToStandard, LuminousFuncToFunc>(tc);
  patterns.add<LuminousModuleToStandard>(module.getContext());
  FrozenRewritePatternSet frozen(std::move(patterns));
  return applyPartialConversion(module, target, frozen);
}

struct LuminousToStandard
    : public ConvertLuminousToStandardBase<LuminousToStandard> {
  void runOnOperation() override {
    if (failed(applyPatterns(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertLuminousToStandardPass() {
  return std::make_unique<LuminousToStandard>();
}
