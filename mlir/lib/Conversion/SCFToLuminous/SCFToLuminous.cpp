//===- SCFToLuminous.cpp - Convert loops to Luminous kernels --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a straightforward conversion of a loop nest into a Luminous
// kernel.  The caller is expected to guarantee that the conversion is correct
// or to further transform the kernel to ensure correctness.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToLuminous/SCFToLuminous.h"
#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "loops-to-luminous"

static constexpr StringLiteral kVisitedAttrName = "SCFToLuminous_visited";

using namespace mlir;
using namespace mlir::scf;

namespace {
struct ParallelToLuminousLowering : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace

/// Lower a `scf.parallel` operation into a corresponding `luminous.dispatch`
/// operation.
///
/// This essentially transforms a loop nest into a corresponding SIMT function.
/// The conversion is driven by mapping annotations on the `scf.parallel`
/// operations. The mapping is provided via a `DictionaryAttribute` named
/// `mapping`, which has three entries:
///  - processor: the hardware id to map to. 0-2 are block dimensions, 3-5 are
///               thread dimensions and 6 is sequential.
///  - map : An affine map that is used to pre-process hardware ids before
///          substitution.
///  - bound : An affine map that is used to compute the bound of the hardware
///            id based on an upper bound of the number of iterations.
/// If the `scf.parallel` contains nested `scf.parallel` operations, those
/// need to be annotated, as well. Structurally, the transformation works by
/// splicing all operations from nested `scf.parallel` operations into a single
/// sequence. Indices mapped to hardware ids are substituted with those ids,
/// wheras sequential mappings result in a sequential for-loop. To have more
/// flexibility when mapping code to hardware ids, the transform supports two
/// affine maps. The first `map` is used to compute the actual index for
/// substitution from the hardware id. The second `bound` is used to compute the
/// launch dimension for the hardware id from the number of iterations the
/// mapped loop is performing. Note that the number of iterations might be
/// imprecise if the corresponding loop-bounds are loop-dependent. In such case,
/// the hardware id might iterate over additional indices. The transformation
/// caters for this by predicating the created sequence of instructions on
/// the actual loop bound. This only works if an static upper bound for the
/// dynamic loop bound can be derived, currently via analyzing `affine.min`
/// operations.
LogicalResult
ParallelToLuminousLowering::matchAndRewrite(ParallelOp parallelOp,
                                            PatternRewriter &rewriter) const {
  // Mark the operation as visited for recursive legality check.
  parallelOp->setAttr(kVisitedAttrName, rewriter.getUnitAttr());

  // We can only transform starting at the outer-most loop. Launches inside of
  // parallel loops are not supported.
  if (auto parentLoop = parallelOp->getParentOfType<ParallelOp>())
    return failure();
  // Create a launch operation. We start with bound one for all grid/block
  // sizes. Those will be refined later as we discover them from mappings.
  Location loc = parallelOp.getLoc();
  Value constantOne = rewriter.create<ConstantIndexOp>(parallelOp.getLoc(), 1);
  luminous::DispatchOp dispatchOp = rewriter.create<luminous::DispatchOp>(
      parallelOp.getLoc(),
      )
  gpu::LaunchOp launchOp = rewriter.create<gpu::LaunchOp>(
      parallelOp.getLoc(), constantOne, constantOne, constantOne, constantOne,
      constantOne, constantOne);
  rewriter.setInsertionPointToEnd(&launchOp.body().front());
  rewriter.create<gpu::TerminatorOp>(loc);
  rewriter.setInsertionPointToStart(&launchOp.body().front());

  BlockAndValueMapping cloningMap;
  llvm::DenseMap<gpu::Processor, Value> launchBounds;
  SmallVector<Operation *, 16> worklist;
  if (failed(processParallelLoop(parallelOp, launchOp, cloningMap, worklist,
                                 launchBounds, rewriter)))
    return failure();

  // Whether we have seen any side-effects. Reset when leaving an inner scope.
  bool seenSideeffects = false;
  // Whether we have left a nesting scope (and hence are no longer innermost).
  bool leftNestingScope = false;
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    // Now walk over the body and clone it.
    // TODO: This is only correct if there either is no further scf.parallel
    //       nested or this code is side-effect free. Otherwise we might need
    //       predication. We are overly conservative for now and only allow
    //       side-effects in the innermost scope.
    if (auto nestedParallel = dyn_cast<ParallelOp>(op)) {
      // Before entering a nested scope, make sure there have been no
      // sideeffects until now.
      if (seenSideeffects)
        return failure();
      // A nested scf.parallel needs insertion of code to compute indices.
      // Insert that now. This will also update the worklist with the loops
      // body.
      if (failed(processParallelLoop(nestedParallel, launchOp, cloningMap,
                                     worklist, launchBounds, rewriter)))
        return failure();
    } else if (op == launchOp.getOperation()) {
      // Found our sentinel value. We have finished the operations from one
      // nesting level, pop one level back up.
      auto parent = rewriter.getInsertionPoint()->getParentOp();
      rewriter.setInsertionPointAfter(parent);
      leftNestingScope = true;
      seenSideeffects = false;
    } else {
      // Otherwise we copy it over.
      Operation *clone = rewriter.clone(*op, cloningMap);
      cloningMap.map(op->getResults(), clone->getResults());
      // Check for side effects.
      // TODO: Handle region side effects properly.
      seenSideeffects |= !MemoryEffectOpInterface::hasNoEffect(clone) ||
                         clone->getNumRegions() != 0;
      // If we are no longer in the innermost scope, sideeffects are disallowed.
      if (seenSideeffects && leftNestingScope)
        return failure();
    }
  }

  // Now that we succeeded creating the launch operation, also update the
  // bounds.
  for (auto bound : launchBounds)
    launchOp.setOperand(getLaunchOpArgumentNum(std::get<0>(bound)),
                        std::get<1>(bound));

  rewriter.eraseOp(parallelOp);
  return success();
}
