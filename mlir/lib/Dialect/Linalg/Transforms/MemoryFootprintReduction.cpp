//===- MemoryFootprintReduction.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the linalg dialect memory footprint reduction pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/Analysis/MemoryFootprintReductionAnalysis.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::scf;
using namespace mlir::luminous;

namespace {

// Parsa: This class is a copied and modified version of
// CollapseSingleIterationLoops rewrite pattern in mlir/lib/Dialect/SCF/SCF.cpp
struct AttrPropagatingSingleIterationLoopCanonicalizer
    : public OpRewritePattern<ParallelOp> {
  // Collapse loop dimensions that perform a single iteration.
  using OpRewritePattern<ParallelOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ParallelOp op,
                                PatternRewriter &rewriter) const override {

    // Parsa: Only perform this canonicalization on loops with specified
    // attribute
    if (!op->hasAttr(launchAttrName))
      return success();

    BlockAndValueMapping mapping;
    // Compute new loop bounds that omit all single-iteration loop dimensions.
    SmallVector<Value, 2> newLowerBounds;
    SmallVector<Value, 2> newUpperBounds;
    SmallVector<Value, 2> newSteps;
    newLowerBounds.reserve(op.lowerBound().size());
    newUpperBounds.reserve(op.upperBound().size());
    newSteps.reserve(op.step().size());
    for (auto dim : llvm::zip(op.lowerBound(), op.upperBound(), op.step(),
                              op.getInductionVars())) {
      Value lowerBound, upperBound, step, iv;
      std::tie(lowerBound, upperBound, step, iv) = dim;
      // Collect the statically known loop bounds.
      auto lowerBoundConstant =
          dyn_cast_or_null<ConstantIndexOp>(lowerBound.getDefiningOp());
      auto upperBoundConstant =
          dyn_cast_or_null<ConstantIndexOp>(upperBound.getDefiningOp());
      auto stepConstant =
          dyn_cast_or_null<ConstantIndexOp>(step.getDefiningOp());
      // Replace the loop induction variable by the lower bound if the loop
      // performs a single iteration. Otherwise, copy the loop bounds.
      if (lowerBoundConstant && upperBoundConstant && stepConstant &&
          (upperBoundConstant.getValue() - lowerBoundConstant.getValue()) > 0 &&
          (upperBoundConstant.getValue() - lowerBoundConstant.getValue()) <=
              stepConstant.getValue()) {
        mapping.map(iv, lowerBound);
      } else {
        newLowerBounds.push_back(lowerBound);
        newUpperBounds.push_back(upperBound);
        newSteps.push_back(step);
      }
    }

    // Parsa: I removed the stuff for reduction loops, we won't have reductions
    // Exit if none of the loop dimensions perform a single iteration.
    if (newLowerBounds.empty() ||
        (newLowerBounds.size() == op.lowerBound().size()))
      return failure();

    // Replace the parallel loop by lower-dimensional parallel loop.
    auto newOp =
        rewriter.create<ParallelOp>(op.getLoc(), newLowerBounds, newUpperBounds,
                                    newSteps, op.initVals(), nullptr);
    // Parsa: Propagate our attribute
    newOp->setAttr(launchAttrName, op->getAttr(launchAttrName));
    // Clone the loop body and remap the block arguments of the collapsed loops
    // (inlining does not support a cancellable block argument mapping).
    rewriter.cloneRegionBefore(op.region(), newOp.region(),
                               newOp.region().begin(), mapping);
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

/// Parsa: This class uses linang tiling rewrite pattern to tile the linalg op,
/// then adds an attribute specifying it's maximum memory footprint to the
/// generated loops; Further it adds an attribute to the reduced linang op,
/// specifying that it is ready for dispatch
template <typename OpTy>
struct MemReductionLinalgTilingPattern : public LinalgBaseTilingPattern {
  const int64_t maxMemFootprint;

  MemReductionLinalgTilingPattern(
      int64_t maxFootprint, MLIRContext *context, LinalgTilingOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : LinalgBaseTilingPattern(OpTy::getOperationName(), context, options,
                                filter, benefit),
        maxMemFootprint(maxFootprint) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    TiledLinalgOp tiledLinalgOp;
    if (failed(LinalgBaseTilingPattern::matchAndRewriteBase(op, rewriter,
                                                            tiledLinalgOp)))
      return failure();

    for (auto *loop : tiledLinalgOp.loops)
      loop->setAttr(launchAttrName, rewriter.getUnitAttr());

    tiledLinalgOp.op->setAttr(maxMemoryAttrName,
                              rewriter.getI64IntegerAttr(maxMemFootprint));
    if (tiledLinalgOp.tensorResults.empty())
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, tiledLinalgOp.tensorResults);

    return success();
  }
};

// Parsa: Copied from mlir/lib/Dialect/Linalg/Tiling.cpp
/// Helper classes for type list expansion.
template <typename... OpTypes>
class RewritePatternList;

template <>
class RewritePatternList<> {
public:
  static void insert(int64_t maxMemoryFootprint, RewritePatternSet &patterns,
                     const LinalgTilingOptions &options) {}
};

template <typename OpTy, typename... OpTypes>
class RewritePatternList<OpTy, OpTypes...> {
public:
  static void insert(int64_t maxMemoryFootprint, RewritePatternSet &patterns,
                     const LinalgTilingOptions &options) {
    auto *ctx = patterns.getContext();
    patterns.add<MemReductionLinalgTilingPattern<OpTy>>(
        maxMemoryFootprint, ctx, options,
        LinalgTransformationFilter(ArrayRef<Identifier>{},
                                   Identifier::get("tiled", ctx)));
    RewritePatternList<OpTypes...>::insert(maxMemoryFootprint, patterns,
                                           options);
  }
};

// Parsa: Copied from mlir/lib/Dialect/Linalg/Tiling.cpp
// Got rid of PadTensorOpTilingPattern
/// Populate the given list with patterns that apply Linalg tiling.
static void insertTilingPatterns(int64_t maxMemoryFootprint,
                                 RewritePatternSet &patterns,
                                 const LinalgTilingOptions &options) {
  RewritePatternList<GenericOp,
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                     >::insert(maxMemoryFootprint, patterns, options);
}

static void
applyTilingToLoopPatterns(int64_t maxMemoryFootprint, FuncOp funcOp,
                          TileSizeComputationFunction tileCompFunc,
                          ArrayRef<StringRef> distributionTypes = {}) {
  auto options = LinalgTilingOptions()
                     .setTileSizeComputationFunction(tileCompFunc)
                     .setLoopType(LinalgTilingLoopType::ParallelLoops)
                     .setDistributionTypes(distributionTypes);
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);
  insertTilingPatterns(maxMemoryFootprint, patterns, options);
  patterns.add<AttrPropagatingSingleIterationLoopCanonicalizer>(ctx);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  (void)applyPatternsAndFoldGreedily(
      funcOp, getLinalgTilingCanonicalizationPatterns(ctx));
  // Drop the marker.
  funcOp.walk([](LinalgOp op) {
    op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}

struct LinalgMemoryFootprintReductionPass
    : public LinalgMemoryFootprintReductionBase<
          LinalgMemoryFootprintReductionPass> {
  MemReduceFn reduceFn;
  LinalgMemoryFootprintReductionPass() = default;
  LinalgMemoryFootprintReductionPass(int64_t maxFootprint, MemReduceFn fn)
      : reduceFn(fn) {
    maxMemFootprint = maxFootprint;
  }

  void runOnFunction() override {
    // Apply tiling patterns for each linalg op here
    if (maxMemFootprint <= 0)
      return;

    auto tileCompFunc = [this](OpBuilder &builder,
                               Operation *op) -> SmallVector<Value, 4> {
      auto theOp = dyn_cast<LinalgOp>(op);
      if (!theOp)
        return {};

      // Get the appropriate tiling shape for this generic op, map them to mlir
      // value
      return llvm::to_vector<4>(llvm::map_range(
          computeTileSizesForMemoryFootprintReduction(theOp, maxMemFootprint,
                                                      reduceFn),
          [&](int64_t s) -> Value {
            return builder.create<ConstantIndexOp>(op->getLoc(), s);
          }));
    };

    applyTilingToLoopPatterns(maxMemFootprint, this->getFunction(),
                              tileCompFunc);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgMemoryFootprintReductionPass(int64_t maxFootprint,
                                               MemReduceFn fn) {
  return std::make_unique<LinalgMemoryFootprintReductionPass>(maxFootprint, fn);
}