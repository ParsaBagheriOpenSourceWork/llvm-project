//===- OpExtraction.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

static SmallVector<Value, 2>
getOperands(MutableArrayRef<OpOperand> opOperands) {
  return llvm::to_vector<2>(llvm::map_range(
      opOperands, [](OpOperand &opOperand) { return opOperand.get(); }));
}

static FuncOp createFunction(Operation *op, ModuleOp module,
                             PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto args = llvm::to_vector<2>(
      llvm::map_range(op->getOpOperands(), [](OpOperand &opOperand) {
        return opOperand.get().getType();
      }));
//  auto module = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));

  auto fnType = rewriter.getFunctionType(args, {});
  SmallString<32> fnName("__extracetd_");
  fnName += op->getName().getStringRef();
  return rewriter.create<FuncOp>(op->getLoc(), fnName, fnType);
}

static ModuleOp createModule(Operation *op, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto thisModule = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(thisModule.getBody(),
                             std::prev(thisModule.getBody()->end()));
  return rewriter.create<ModuleOp>(op->getLoc());
}

namespace mlir {
namespace linalg {

template <typename OpTy>
LogicalResult ExtractOpRewritePattern<OpTy>::matchAndRewrite(
    OpTy op, mlir::PatternRewriter &rewriter) const {
  auto newModule = createModule(op, rewriter);
  auto fn = createFunction(op, newModule, rewriter);
  auto cloned = op->cloneWithoutRegions();

  rewriter.inlineRegionBefore(op.getRegion(), cloned->getRegion(0),
                              cloned->getRegion(0).begin());
  rewriter.replaceOpWithNewOp<mlir::CallOp>(op, fn,
                                            getOperands(op->getOpOperands()));
  auto block = fn.addEntryBlock();
  block->push_back(cloned);
  return mlir::success();
}

} // namespace linalg
} // namespace mlir

namespace {

struct LinalgOpExtractionPass
    : public LinalgOpExtractionBase<LinalgOpExtractionPass> {

  void runOnFunction() override {
    auto funcOp = getOperation();
    auto *cntx = funcOp.getContext();
    mlir::RewritePatternSet patterns(cntx);
    patterns.add<ExtractOpRewritePattern<GenericOp>>(cntx);
    mlir::FrozenRewritePatternSet frozen(std::move(patterns));
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(funcOp, frozen))) {
      llvm::errs() << "Failed to apply the rewrite pattern" << funcOp.getLoc()
                   << "\n";
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgOpExtractionPass() {
  return std::make_unique<LinalgOpExtractionPass>();
}
