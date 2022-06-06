//===- SCFToLuminous.cpp - SCF to Luminous conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert scf.parallel operations into luminous
// dispatches. This file contains copies of some static functions and types
// from mlir/lib/Dialect/Async/Transform/AsyncParallelFor.cpp
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToLuminous/SCFToLuminous.h"
#include "../PassDetail.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::async;
using namespace mlir::luminous;
using namespace mlir::linalg;

namespace {

struct LuminousDispatchParallelRewrite
    : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override;
};

/// Applies the conversion patterns in the given function.
static LogicalResult applyPatterns(ModuleOp module) {
  ConversionTarget target(*module.getContext());

  // parallel ops that still have launch attribute and
  // parallel loops inside launch ops are illegal
  target.addDynamicallyLegalOp<scf::ParallelOp>([](Operation *op) {
    return !op->hasAttr(launchAttrName) && !op->getParentOfType<LaunchOp>();
  });
  target.addLegalDialect<LuminousDialect, LinalgDialect, AsyncDialect>();
  RewritePatternSet patterns(module.getContext());
  patterns.add<LuminousDispatchParallelRewrite>(module.getContext());
  FrozenRewritePatternSet frozen(std::move(patterns));
  return applyPartialConversion(module, target, frozen);
}

/// A pass converting parallel loops with luminous launch attribute
/// to Luminous operations.
struct SCFToLuminousPass
    : public ConvertParallelLoopToLuminousLaunchBase<SCFToLuminousPass> {
  /// Pass entry point.
  void runOnOperation() override {
    if (failed(applyPatterns(getOperation())))
      signalPassFailure();
  }
};

LogicalResult LuminousDispatchParallelRewrite::matchAndRewrite(
    scf::ParallelOp op, PatternRewriter &rewriter) const {

  // Only perform this conversion on attributed parallel ops
  if (!op->hasAttr(launchAttrName))
    return failure();

  if (op.getNumReductions() != 0)
    return failure();

  auto launchOp = rewriter.replaceOpWithNewOp<LaunchOp>(op, op.getUpperBound(),
                                                        op.getStep());
  {
    OpBuilder::InsertionGuard innerGuard(rewriter);
    rewriter.setInsertionPointToEnd(op.getBody());
    assert(llvm::hasSingleElement(op.getRegion()) &&
           "expected scf.parallel to have one block");
    rewriter.replaceOpWithNewOp<luminous::YieldOp>(
        op.getBody()->getTerminator());
  }
  rewriter.inlineRegionBefore(op.getLoopBody(), launchOp.body(),
                              launchOp.body().begin());
  return success();
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertParallelForToLuminousLaunchPass() {
  return std::make_unique<SCFToLuminousPass>();
}