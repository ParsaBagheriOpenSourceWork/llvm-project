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

static FuncOp createFunction(Operation *op, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto args = llvm::to_vector<2>(
      llvm::map_range(op->getOpOperands(), [](OpOperand &opOperand) {
        return opOperand.get().getType();
      }));
  auto module = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));

  auto fnType = rewriter.getFunctionType(args, {});
  SmallString<32> fnName("__extracetd_");
  fnName += op->getName().getStringRef();
  //  return rewriter.create<FuncOp>(op->getLoc(), fnName, fnType);
  auto fn = rewriter.create<FuncOp>(op->getLoc(), fnName, fnType);
  auto *block = fn.addEntryBlock();
  rewriter.setInsertionPoint(block, std::prev(block->end()));
  rewriter.create<ReturnOp>(op->getLoc());
  fn->setAttr("dispatch", StringAttr::get(op->getContext(), "luminous.func"));
  return fn;
}

namespace mlir {
namespace linalg {

// TODO: Add typetrait to restrict this to Operation types
template <typename OpTy>
LogicalResult ExtractOpRewritePattern<OpTy>::matchAndRewrite(
    OpTy op, mlir::PatternRewriter &rewriter) const {
  auto fn = createFunction(op, rewriter);
  auto cloned = op->cloneWithoutRegions();
  rewriter.inlineRegionBefore(op.getRegion(), cloned->getRegion(0),
                              cloned->getRegion(0).begin());
  auto &block = fn.back();
  block.push_front(cloned);
  auto call = rewriter.replaceOpWithNewOp<mlir::CallOp>(
      op, fn, getOperands(op->getOpOperands()));
  call->setAttr("dispatch", StringAttr::get(op->getContext(), "luminous.call"));
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
