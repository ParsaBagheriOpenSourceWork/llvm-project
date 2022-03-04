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

#include "../lib/Dialect/Async/Transforms/PassDetail.h" /* cloneConstantsIntoTheRegion */
#include "PassDetail.h"
#include "mlir/Conversion/SCFToLuminous/SCFToLuminous.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"
#include "mlir/Dialect/Luminous/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::async;
using namespace mlir::luminous;
using namespace mlir::linalg;

static constexpr char luminousModuleSymbol[] = "device_module";
static constexpr char kernelOutliningVisited[] = "kerenels_outlined";

static const std::string getNextFunctionName() {
  static constexpr char luminousAsyncFnSymbol[] = "async_fn";
  static unsigned fnId = 0;
  return luminousAsyncFnSymbol + std::to_string(fnId++);
}

namespace mlir {
namespace luminous {

/// This class handles extracting and rewriting dispatchable regions within
/// LaunchOps.
/// I) Initial state: A basic block is constructed. At this point operations
/// from the target LaunchOp can be inserted to the block.
/// II) FunctionCreated state: After the basic block is filled with specified
/// operations, a luminous function is created that will wrap the basic block.
/// III) DispatchCreated state: The operations in the launch op corresponding
/// to the newly created luminous function are replaced with a luminous
/// dispatch call.
class DispatchableBlock {
  friend class DispatchableBlocks;
  friend void outlineKernels(PatternRewriter &rewriter, Location loc,
                             LuminousModuleOp luminousModule,
                             DispatchableBlocks &&dispatchBlocks);

  OpBuilder builder;
  LaunchOp launchOp;
  std::unique_ptr<Block> block;
  SmallVector<Value> args;
  SmallVector<Operation *> ops;
  SetVector<Value> opResults;
  BlockAndValueMapping cloningMap;

  DispatchableBlock(LaunchOp op)
      : builder(op.getContext()), launchOp(op), block(new Block) {}
  DispatchableBlock(DispatchableBlock &&other) = default;
  DispatchableBlock &operator=(DispatchableBlock &&other) = default;

  Block *releaseBlock() { return block.release(); }

public:
  DispatchableBlock(const DispatchableBlock &) = delete;
  DispatchableBlock &operator=(const DispatchableBlock &other) = delete;

  /// Clones `op' and inserts it in the basic block; it keeps track of
  /// the op for performing replacement with dispatch call.
  void push_back(Operation *op) {
    assert(op->getParentOp() == launchOp && "can only push back operations "
                                            "within launch op");
    // handling operands
    for (auto operand : op->getOperands()) {
      // if the operands of the current op have been visited before then
      // continue, otherwise they are arguments to this block.
      if (opResults.contains(operand))
        continue;

      args.push_back(operand);
      auto blockArg = block->addArgument(operand.getType(), operand.getLoc());
      cloningMap.map(operand, blockArg);
    }

    // keeping track of ops results to determine whether the value is an
    // argument to the block or has been produced in the block
    for (auto result : op->getOpResults()) {
      assert(!opResults.contains(result));
      opResults.insert(result);
    }

    // keeping track of ops to later replace them with a dispatch call
    ops.push_back(op);
    auto *clone = builder.clone(*op, cloningMap);
    block->push_back(clone);
  }
};

/// Encapsulates the dispatchable blocks and provides an interface to add new
/// dispatchable blocks
class DispatchableBlocks {
  friend void outlineKernels(PatternRewriter &rewriter, Location loc,
                             LuminousModuleOp luminousModule,
                             DispatchableBlocks &&dispatchBlocks);
  LaunchOp launchOp;

  using Blocks = SmallVector<std::unique_ptr<DispatchableBlock>>;
  Blocks blocks;

  Blocks &getBlocks() { return blocks; }

public:
  DispatchableBlocks(LaunchOp op) : launchOp(op) {}
  DispatchableBlock &addNewBlock() {
    blocks.emplace_back(new DispatchableBlock(launchOp));
    return *blocks.back();
  }
};

/// Outlines the luminous function
static LuminousFuncOp outline(PatternRewriter &rewriter, Location loc,
                              LuminousModuleOp luminousModule, Block *block) {

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(luminousModule.getBody()->getTerminator());

  // return type is void and arguments are the basic blocks arguments
  auto funcType =
      rewriter.getFunctionType(block->getArgumentTypes(), TypeRange());

  auto func =
      rewriter.create<LuminousFuncOp>(loc, getNextFunctionName(), funcType);

  rewriter.setInsertionPoint(block, block->end());
  rewriter.create<luminous::ReturnOp>(loc);

  // transferring ownership of the block to the funcs region
  func.body().push_front(block);

  return func;
}

/// Replaces the ops in the target launch op with dispatch calls to the
/// outlined luminous functions
static Value replaceOpsWithDispatchCall(PatternRewriter &rewriter,
                                        LuminousFuncOp func,
                                        ValueRange dependencies,
                                        SmallVector<Operation *> &ops,
                                        SmallVector<Value> &args) {
  OpBuilder::InsertionGuard guard(rewriter);

  // Insert dispatch call after the last op in the dispatchable region
  auto *lastOp = ops.back();
  rewriter.setInsertionPoint(lastOp);
  auto dispatchOp =
      rewriter.create<DispatchOp>(lastOp->getLoc(), func, dependencies, args);

  // removing the dispatchable regions ops from the launch ops body
  for (auto *op : ops)
    rewriter.eraseOp(op);

  return dispatchOp;
}

/// Outlines kernels for each dispatchable blocks
/// this function consumes and destroys dispatchBlocks
void outlineKernels(PatternRewriter &rewriter, Location loc,
                    LuminousModuleOp luminousModule,
                    DispatchableBlocks &&dispatchBlocks) {
  auto &blocks = dispatchBlocks.getBlocks();
  Value dispOpToken = nullptr;
  for (auto &block : blocks) {
    auto fn = outline(rewriter, loc, luminousModule, block->releaseBlock());
    if (dispOpToken)
      dispOpToken = replaceOpsWithDispatchCall(rewriter, fn, {dispOpToken},
                                               block->ops, block->args);
    else
      dispOpToken =
          replaceOpsWithDispatchCall(rewriter, fn, {}, block->ops, block->args);
  }

  if (dispOpToken) {
    ImplicitLocOpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(dispOpToken);
    rewriter.create<AwaitOp>(loc, dispOpToken);
  }
}

static LuminousModuleOp findLuminousModule(ModuleOp module) {
  auto *op = module.lookupSymbol(luminousModuleSymbol);
  assert(op && isa<LuminousModuleOp>(op) && "couldn't find luminous module");
  return cast<LuminousModuleOp>(op);
}

static LuminousModuleOp createLuminousModule(PatternRewriter &rewriter,
                                             Location loc, ModuleOp module) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(module.getBody(), module.getBody()->begin());
  MLIRContext *ctx = module.getContext();
  auto luminousModule =
      rewriter.create<LuminousModuleOp>(loc, luminousModuleSymbol);
  module->setAttr(LuminousDialect::getContainerModuleAttrName(),
                  UnitAttr::get(ctx));
  return luminousModule;
}

/// Finds the LuminousModuleOp in `op's parent module if it exists,
/// otherwise creates a new LuminousModuleOp.
static LuminousModuleOp getLuminousModule(Operation *op, Location loc,
                                          PatternRewriter &rewriter) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (module->hasAttr(LuminousDialect::getContainerModuleAttrName()))
    return findLuminousModule(module);

  return createLuminousModule(rewriter, loc, module);
}

void defaultDispatchBuilderFn(LaunchOp launchOp,
                              DispatchableBlocks &dispatchableBlocks) {
  auto &body = launchOp.body().back();
  for (auto &op : body) {
    if (!op.hasAttr(luminous::maxMemoryAttrName))
      continue;

    // add a new block to the dispatchable blocks and fill it with ops
    auto &block = dispatchableBlocks.addNewBlock();
    block.push_back(&op);
  }
}

} // namespace luminous
} // namespace mlir

namespace {

struct KernelOutliningRewritePattern : public OpRewritePattern<LaunchOp> {
  DispatchBuilderFn dispatchBuilderFn;
  KernelOutliningRewritePattern(MLIRContext *ctx, DispatchBuilderFn fn)
      : OpRewritePattern<LaunchOp>(ctx), dispatchBuilderFn(fn) {}
  LogicalResult matchAndRewrite(LaunchOp op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult applyPatterns(FuncOp funcOp, DispatchBuilderFn fn) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<KernelOutliningRewritePattern>(ctx, fn);
  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

/// A pass converting SCF operations to OpenMP operations.
struct LuminousKernelOutliningPass
    : public LuminousKernelOutliningBase<LuminousKernelOutliningPass> {
  DispatchBuilderFn dispatchBuilderFn;

  LuminousKernelOutliningPass(DispatchBuilderFn fn)
      : dispatchBuilderFn(std::move(fn)) {}

  /// Pass entry point.
  void runOnOperation() override {
    if (failed(applyPatterns(getOperation(), dispatchBuilderFn)))
      signalPassFailure();
  }
};

LogicalResult KernelOutliningRewritePattern::matchAndRewrite(
    LaunchOp op, PatternRewriter &rewriter) const {

  // stop if already visited
  if (op->hasAttr(kernelOutliningVisited))
    return failure();

  auto loc = op.getLoc();
  auto luminousModule = getLuminousModule(op, loc, rewriter);
  op->setAttr(kernelOutliningVisited, rewriter.getUnitAttr());
  DispatchableBlocks dispatchableBlocks(op);
  dispatchBuilderFn(op, dispatchableBlocks);
  outlineKernels(rewriter, loc, luminousModule, std::move(dispatchableBlocks));

  return success();
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLuminousKernelOutliningPass(DispatchBuilderFn fn) {
  return std::make_unique<LuminousKernelOutliningPass>(fn);
}
