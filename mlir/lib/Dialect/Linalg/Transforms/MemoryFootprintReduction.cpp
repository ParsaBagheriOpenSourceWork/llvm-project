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
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::linalg;

constexpr const char *reductionAttrName = "max-memory-footprint";

/// Uses linang tiling rewrite pattern to tile the linalg op,
/// then adds an attribute specifying it's maximum memory footprint to the
/// generated loops
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

    rewriter.
    for (auto *loop : tiledLinalgOp.loops) {
      llvm::errs() << "before\n";
      loop->dump();
      loop->template walk([&](scf::ParallelOp parallelOp) {
        for (auto dim :
             llvm::zip(parallelOp.lowerBound(), parallelOp.upperBound())) {
          if (std::get<0>(dim) == std::get<1>(dim)) {
            rewriter.replaceOp(parallelOp, parallelOp.initVals());
          }
        }
      });
      llvm::errs() << "after\n";
      loop->dump();
      loop->setAttr(reductionAttrName,
                    rewriter.getI64IntegerAttr(maxMemFootprint));
    }

    if (tiledLinalgOp.tensorResults.empty())
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, tiledLinalgOp.tensorResults);

    return success();
  }
};

// Copied from mlir/lib/Dialect/Linalg/Tiling.cpp
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

// Copied from mlir/lib/Dialect/Linalg/Tiling.cpp
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
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  (void)applyPatternsAndFoldGreedily(
      funcOp, getLinalgTilingCanonicalizationPatterns(ctx));
  // Drop the marker.
  funcOp.walk([](LinalgOp op) {
    op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}

namespace {
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
