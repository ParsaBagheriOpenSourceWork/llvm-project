//===- LinalgToLuminous.cpp - Linalg to Luminous conversion ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert linalg operations into luminous
// launches.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LinalgToLuminous/LinalgToLuminous.h"
#include "../PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/Analysis/MemoryFootprintReductionAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::luminous;
using namespace async;

namespace {

struct LuminousLaunchLinalgRewrite
    : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {

    // Only perform this conversion on attributed linalg ops that are not in a
    // launch capsule
    if (!linalgOp->hasAttr(luminous::maxMemoryAttrName) ||
        isa<LaunchOp>(linalgOp->getParentOp()) ||
        linalgOp->getParentOp()->hasAttr(luminous::launchAttrName))
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(linalgOp);
    auto shape = LinalgOpShape::create(linalgOp);
    llvm::SetVector<int64_t> uniqueIndices;
    for (auto s : shape.get())
      uniqueIndices.insert(s);

    auto indexValues = llvm::to_vector<4>(
        llvm::map_range(uniqueIndices, [&](int64_t s) -> Value {
          return rewriter.create<arith::ConstantIndexOp>(linalgOp.getLoc(), s);
        }));

    LaunchOp launchOp = rewriter.create<LaunchOp>(
        linalgOp.getLoc(), /* shape */ indexValues, /* step */ indexValues);

    // constructing the block, ownership will be transferred to launchOps region
    Block *block = new Block;
    for (int i = 0, end = launchOp.shape().size(); i < end; i++)
      block->addArgument(mlir::IndexType::get(getContext()), linalgOp.getLoc());

    block->push_back(linalgOp->clone());
    rewriter.setInsertionPointToEnd(block);
    rewriter.create<mlir::luminous::YieldOp>(linalgOp.getLoc());
    launchOp.body().push_back(block);
    rewriter.eraseOp(linalgOp);
    return success();
  }
};

static LogicalResult applyPatterns(ModuleOp module) {
  ConversionTarget target(*module.getContext());
  target.addLegalDialect<LuminousDialect>();
  // marking all linalg ops illegal(will trigger conversion), unless they
  // satisfy lambda
  target.addDynamicallyLegalOp<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >([&](Operation *op) {
    return !op->hasAttr(luminous::maxMemoryAttrName) ||
           isa<LaunchOp>(op->getParentOp()) ||
           op->getParentOp()->hasAttr(luminous::launchAttrName);
  });
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  RewritePatternSet patterns(module.getContext());
  patterns.add<LuminousLaunchLinalgRewrite>(module.getContext());
  FrozenRewritePatternSet frozen(std::move(patterns));
  return applyPartialConversion(module, target, frozen);
}

} // namespace

struct LinalgToLuminous
    : public ConvertLinalgToLuminousLaunchBase<LinalgToLuminous> {

  void runOnOperation() override {
    if (failed(applyPatterns(getOperation())))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertLinalgToLuminousLaunchPass() {
  return std::make_unique<LinalgToLuminous>();
}
