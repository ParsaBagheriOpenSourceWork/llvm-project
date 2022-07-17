//
// Created by parsab on 6/27/22.
//

#include "mlir/Conversion/LuminousToStandard/LuminousToStandard.h"
#include "../PassDetail.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace luminous;

namespace {

struct LuminousFuncToFunc : public OpRewritePattern<LuminousFuncOp> {
  using OpRewritePattern<LuminousFuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LuminousFuncOp op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult
LuminousFuncToFunc::matchAndRewrite(LuminousFuncOp op,
                                    PatternRewriter &rewriter) const {
  auto funcOp = rewriter.replaceOpWithNewOp<func::FuncOp>(op, op.getName(),
                                                          op.getFunctionType());
  OpBuilder::InsertionGuard guard(rewriter);
  {
    OpBuilder::InsertionGuard innerGuard(rewriter);
    rewriter.setInsertionPointToEnd(&op.getBody().back());
    assert(llvm::hasSingleElement(op.getRegion()) &&
           "expected luminous.func to have one block");
    rewriter.replaceOpWithNewOp<func::ReturnOp>(
        op.getBody().back().getTerminator());
  }
  rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(),
                              funcOp.getBody().begin());
  funcOp->setAttr(LuminousDialect::getLuminousFuncAttrName(),
                  rewriter.getUnitAttr());
  //  op->getParentOfType<ModuleOp>().dump();
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
  moduleOp->setAttr(LuminousDialect::getLuminousModuleAttrName(),
                    rewriter.getUnitAttr());
  return success();
}

static LogicalResult applyPatterns(ModuleOp module) {
  ConversionTarget target(*module.getContext());
  target.addLegalDialect<LuminousDialect, linalg::LinalgDialect,
                         async::AsyncDialect>();
  target.addLegalDialect<BuiltinDialect, func::FuncDialect>();
  target.addIllegalOp<LuminousFuncOp, LuminousModuleOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  RewritePatternSet patterns(module.getContext());
  patterns.add<LuminousFuncToFunc, LuminousModuleToStandard>(
      module.getContext());
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