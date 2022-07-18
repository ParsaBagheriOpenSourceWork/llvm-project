//===- LuminousToLLVM.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "mlir/Conversion/LuminousToLLVM/LuminousToLLVM.h"
#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace luminous;

namespace {

constexpr char LuminousRuntimeDispatchFn[] = "__luminous_runtime_dispatch";
constexpr char LuminousRuntimeLoadFn[] = "__luminous_runtime_load_capsule";

static LLVM::LLVMStructType
createArgStruct(PatternRewriter &rewriter, OperandRange operands, ValueRange values,
                std::vector<Type> &paramTypes, std::vector<Value> &actuals,
                TypeConverter *typeConverter, StringRef name) {
  for (auto operand_tuple : llvm::zip(operands, values)) {
    auto operand = std::get<0>(operand_tuple);
    if (operand.getType().isa<MemRefType>()) {
      llvm::SmallVector<Value, 5> memrefValues;
      auto operandAdaptor = std::get<1>(operand_tuple);
      MemRefDescriptor::unpack(rewriter, operand.getLoc(), operandAdaptor,
                               operand.getType().cast<MemRefType>(),
                               memrefValues);
      for (auto m : memrefValues) {
        paramTypes.push_back(m.getType());
        actuals.push_back(m);
      }
    } else {
      auto llvmType = typeConverter->convertType(operand.getType());
      assert(llvmType);
      paramTypes.push_back(llvmType);
      actuals.push_back(operand);
    }
  }
  return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), paramTypes,
                                          /* packed */ true);
}

struct LuminousDispatchToLLVM : public ConvertOpToLLVMPattern<DispatchOp> {
  using ConvertOpToLLVMPattern<DispatchOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(DispatchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

static void createWrapperFn(PatternRewriter &rewriter, DispatchOp op,
                            ModuleOp module, LLVM::LLVMStructType paramStruct) {
  OpBuilder::InsertionGuard g(rewriter);

  auto loc = op.getLoc();
  auto llvmInt32Type = IntegerType::get(rewriter.getContext(), 32);
  auto llvmInt8Type = IntegerType::get(rewriter.getContext(), 8);
  auto voidPtrType = LLVM::LLVMPointerType::get(llvmInt8Type);

  auto *luminousModuleOp =
      module.lookupSymbol(op.getFuncModuleName().getValue());
  assert(luminousModuleOp && "luminous module not found");

  auto luminousModule = dyn_cast<ModuleOp>(luminousModuleOp);
  assert(luminousModule && "luminous module must be lowered to builtin module");

  rewriter.setInsertionPointToEnd(luminousModule.getBody(0));
  auto llvmFnUnpacked = LLVM::lookupOrCreateFn(
      luminousModule, op.getFuncName().getValue(), paramStruct.getBody(),
      LLVM::LLVMVoidType::get(op.getContext()));
  auto wrapperName = std::string(op.getFuncName().getValue()) + "_wrapper";
  auto llvmWrapperFn = rewriter.create<LLVM::LLVMFuncOp>(
      loc, wrapperName,
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(op.getContext()),
                                  {voidPtrType}));
  Block *body = new Block;
  body->addArgument(voidPtrType, loc);
  {
    OpBuilder::InsertionGuard innerGuard(rewriter);
    rewriter.setInsertionPointToEnd(body);
    auto zero = rewriter.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                                  rewriter.getI32IntegerAttr(0));
    // Casting void ptr to a pointer to our struct type
    auto argBitCast = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(paramStruct),
        ValueRange(body->getArgument(0)));

    // Unpacking arguments
    std::vector<Value> args;
    for (auto &tuple : llvm::enumerate(paramStruct.getBody())) {
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
}

LogicalResult LuminousDispatchToLLVM::matchAndRewrite(
    DispatchOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  OpBuilder::InsertionGuard g(rewriter);
  auto loc = op.getLoc();
  auto module = op->getParentOfType<ModuleOp>();

  // Types that we need
  auto llvmInt32Type = IntegerType::get(rewriter.getContext(), 32);
  auto llvmInt8Type = IntegerType::get(rewriter.getContext(), 8);
  auto voidPtrType = LLVM::LLVMPointerType::get(llvmInt8Type);

  auto zero = rewriter.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                                rewriter.getI32IntegerAttr(0));
  auto one = rewriter.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                               rewriter.getI32IntegerAttr(1));

  // creating the arguments struct type
  std::vector<Type> paramTypes;
  std::vector<Value> actuals;
  auto paramStruct = createArgStruct(
      rewriter, op.getOperands(), adaptor.getOperands(), paramTypes,
      actuals, typeConverter, op.getFuncName().getValue());

  // instantiating and packing values into an arguments struct
  auto packArgsStruct = [&]() {
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
  };
  auto structPtr = packArgsStruct();

  // creating the string symbol of capsule function
  auto wrapperName = std::string(op.getFuncName().getValue()) + "_wrapper";
  auto createGlobalString = [&]() {
    // using lambda here because I am changing the insertion point
    // and need to use guard to go back to previous insertion point after string
    // is created
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    llvm::SmallString<32> wrapperString(wrapperName);
    wrapperString.push_back('\0'); // Null terminate for C
    size_t stringSize = wrapperString.size_in_bytes();
    auto globalType = LLVM::LLVMArrayType::get(
        llvmInt8Type, stringSize); // add 1 for null terminator
    auto globalString = rewriter.create<LLVM::GlobalOp>(
        loc, globalType,
        /*isConstant=*/true, LLVM::Linkage::Internal, wrapperName,
        rewriter.getStringAttr(wrapperString));
    return globalString;
  };
  auto globalString = createGlobalString();

  auto addrOfString = rewriter.create<LLVM::AddressOfOp>(loc, globalString);
  Value strPtr = rewriter.create<LLVM::GEPOp>(
      loc, voidPtrType, addrOfString.getRes(), ValueRange({zero, zero}));

  // creating runtime calls
  auto runtime_load_fn = LLVM::lookupOrCreateFn(module, LuminousRuntimeLoadFn,
                                                {voidPtrType}, voidPtrType);
  auto loadFnCall =
      rewriter.create<LLVM::CallOp>(loc, runtime_load_fn, ValueRange({strPtr}));
  auto runtime_dispatch_fn =
      LLVM::lookupOrCreateFn(module, LuminousRuntimeDispatchFn,
                             {voidPtrType, voidPtrType, llvmInt32Type},
                             LLVM::LLVMVoidType::get(getContext()));
  auto paramsPtr =
      rewriter.create<LLVM::BitcastOp>(loc, voidPtrType, ValueRange(structPtr));
  DataLayout dl(module);
  auto sizeOfStruct = dl.getTypeSizeInBits(paramStruct);
  auto size = rewriter.create<LLVM::ConstantOp>(
      loc, llvmInt32Type, rewriter.getI32IntegerAttr(sizeOfStruct));
  rewriter.create<LLVM::CallOp>(
      loc, runtime_dispatch_fn,
      ValueRange({loadFnCall.getResult(0), paramsPtr.getRes(), size.getRes()}));

  /// TODO : temporary here ---------------
  // This deletes the async await op. Later we will need to come up with our own
  // async handling runtime
  for (auto user : op->getUsers()) {
    rewriter.eraseOp(user);
  }
  ///--------------------------------------

  // creating a wrapper function for unpacking arguments and calling kernel
  createWrapperFn(rewriter, op, module, paramStruct);
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
  patterns.add<LuminousDispatchToLLVM>(tc);
  FrozenRewritePatternSet frozen(std::move(patterns));
  return applyPartialConversion(module, target, frozen);
}

struct LuminousToLLVM : public ConvertLuminousToLLVMBase<LuminousToLLVM> {
  void runOnOperation() override {
    if (failed(applyPatterns(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertLuminousToLLVMPass() {
  return std::make_unique<LuminousToLLVM>();
}