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

#include "PassDetail.h"
#include "mlir/Conversion/SCFToLuminous/SCFToLuminous.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Luminous/IR/LuminousDialect.h"
#include "mlir/Dialect/Luminous/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::async;
using namespace mlir::luminous;
using namespace mlir::linalg;

constexpr char luminousModuleSymbol[] = "device_module";
constexpr char kernelOutliningVisited[] = "kernels_outlined";

// finds a unique symbol for the 'name' parameter in the luminous module
static std::string getUniqueName(LuminousModuleOp luminousModule,
                                 const std::string &name) {
  auto hasSymbol = [&](const std::string &sym) {
    return luminousModule.lookupSymbol(sym) != nullptr;
  };
  static unsigned id = 0;
  std::string prefix = "async_fn_" + name;
  std::string uniqueName;
  do {
    uniqueName = prefix + std::to_string(id++);
  } while (hasSymbol(uniqueName));
  return uniqueName;
}

namespace mlir {
namespace luminous {

namespace detail {
struct DispatchBlockImpl {
  std::string name;
  OpBuilder builder;
  LaunchOp launchOp;
  std::unique_ptr<Block> block;
  SmallVector<Value> args;
  SmallVector<Operation *> ops;
  SetVector<Value> opResults;
  BlockAndValueMapping cloningMap;
  SmallVector<DispatchBlockImpl *> dependencies;
  DispatchBlockImpl(LaunchOp op, ArrayRef<DispatchBlock> deps,
                    const std::string &name)
      : name(name), builder(op.getContext()), launchOp(op), block(new Block) {
    for (auto d : deps)
      dependencies.push_back(&(d.impl));
  }
  DispatchBlockImpl(const DispatchBlockImpl &) = delete;
  DispatchBlockImpl &operator=(const DispatchBlockImpl &) = delete;
  DispatchBlockImpl(DispatchBlockImpl &&) = default;
  Block *releaseBlock() { return block.release(); }
};

struct DispatchBlocksImpl {
  LuminousModuleOp luminousModule;
  LaunchOp launchOp;
  using Blocks = SmallVector<std::unique_ptr<DispatchBlockImpl>>;
  Blocks blocks;
  DispatchBlocksImpl(LuminousModuleOp module, LaunchOp op)
      : luminousModule(module), launchOp(op) {}
};

} // namespace detail

/// Clones `op' and inserts it in the basic block; it keeps track of
/// the op for performing replacement with dispatch call.
void DispatchBlock::pushBack(Operation *op) {
  assert(op->getParentOp() == impl.launchOp && "can only push back operations "
                                               "within launch op");
  // handling operands
  for (auto operand : op->getOperands()) {
    // if the operands of the current op have been visited before then
    // continue, otherwise they are arguments to this block.
    if (impl.opResults.contains(operand) || impl.cloningMap.contains(operand))
      continue;

    impl.args.push_back(operand);
    auto blockArg =
        impl.block->addArgument(operand.getType(), operand.getLoc());
    impl.cloningMap.map(operand, blockArg);
  }

  // keeping track of ops results to determine whether the value is an
  // argument to the block or has been produced in the block
  for (auto result : op->getOpResults()) {
    assert(!impl.opResults.contains(result));
    impl.opResults.insert(result);
  }

  // keeping track of ops to later replace them with a dispatch call
  impl.ops.push_back(op);
  auto *clone = impl.builder.clone(*op, impl.cloningMap);
  impl.block->push_back(clone);
}

/// Creates a dispatch block implementation in the container and returns a
/// DispatchBlock wrapper of the implementation.
DispatchBlock
DispatchBlocks::addNewBlock(llvm::ArrayRef<DispatchBlock> dependencies,
                            const std::string &name) {
  impl.blocks.emplace_back(new detail::DispatchBlockImpl(
      impl.launchOp, dependencies, getUniqueName(impl.luminousModule, name)));
  return DispatchBlock(*impl.blocks.back());
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

/// Outlines the luminous functions and inserts dispatch calls, dependencies
/// first
static void
outline(PatternRewriter &rewriter, Location loc,
        LuminousModuleOp luminousModule, detail::DispatchBlockImpl *impl,
        llvm::DenseMap<detail::DispatchBlockImpl *, Value> &outlined,
        llvm::SetVector<Value> &usedDispatchToken) {
  if (outlined.find(impl) != outlined.end())
    return;

  for (auto *dep : impl->dependencies)
    outline(rewriter, loc, luminousModule, dep, outlined, usedDispatchToken);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(luminousModule.getBody()->getTerminator());

  // at this point our dependencies should be handled, now we can perform the
  // dispatches
  SmallVector<Value> dependencyTokens;
  for (auto *dep : impl->dependencies) {
    usedDispatchToken.insert(outlined[dep]);
    dependencyTokens.push_back(outlined[dep]);
  }

  auto *block = impl->releaseBlock();
  assert(block && "block is already outlined");

  // return type is void and arguments are the basic blocks arguments
  auto funcType =
      rewriter.getFunctionType(block->getArgumentTypes(), TypeRange());

  auto func = rewriter.create<LuminousFuncOp>(loc, impl->name, funcType);

  rewriter.setInsertionPoint(block, block->end());
  rewriter.create<luminous::ReturnOp>(loc);

  // transferring ownership of the block to the funcs region
  func.body().push_front(block);

  auto dispatchToken = replaceOpsWithDispatchCall(
      rewriter, func, dependencyTokens, impl->ops, impl->args);

  outlined.insert({impl, dispatchToken});
}

/// Outlines kernels for each dispatch blocks
void outlineKernels(LaunchOp op, PatternRewriter &rewriter, Location loc,
                    LuminousModuleOp luminousModule,
                    detail::DispatchBlocksImpl &dispatchBlocks) {
  llvm::DenseMap<detail::DispatchBlockImpl *, Value> outlined;
  llvm::SetVector<Value> usedDispatchTokens;
  for (auto &block : dispatchBlocks.blocks) {
    outline(rewriter, loc, luminousModule, block.get(), outlined,
            usedDispatchTokens);
  }

  ImplicitLocOpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(
      op.body().back().getTerminator()->getPrevNode());
  for (auto &p : outlined) {
    auto dispatchToken = p.second;
    if (!usedDispatchTokens.contains(dispatchToken)) {
      rewriter.create<AwaitOp>(loc, dispatchToken);
    }
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
                              DispatchBlocks &dispatchBlocks) {
  auto &body = launchOp.body().back();
  SmallVector<DispatchBlock> dependencies;
  for (auto &op : body) {
    if (!op.hasAttr(luminous::maxMemoryAttrName))
      continue;

    // add a new block to the dispatch blocks and fill it with ops
    auto block = dispatchBlocks.addNewBlock(dependencies);
    dependencies.clear();
    block.pushBack(&op);
    dependencies.push_back(block);
  }
}

} // namespace luminous
} // namespace mlir

namespace {

struct KernelOutliningRewritePattern : public OpRewritePattern<LaunchOp> {
  DispatchBuilderFn dispatchBuilderFn;
  KernelOutliningRewritePattern(MLIRContext *ctx, DispatchBuilderFn fn)
      : OpRewritePattern<LaunchOp>(ctx), dispatchBuilderFn(std::move(fn)) {}
  LogicalResult matchAndRewrite(LaunchOp op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult applyPatterns(func::FuncOp funcOp, const DispatchBuilderFn &fn) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<KernelOutliningRewritePattern>(ctx, fn);
  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

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
  mlir::luminous::detail::DispatchBlocksImpl dispatchBlocksImpl(luminousModule,
                                                                op);
  DispatchBlocks dispatchBlocks(dispatchBlocksImpl);
  dispatchBuilderFn(op, dispatchBlocks);
  outlineKernels(op, rewriter, loc, luminousModule, dispatchBlocksImpl);

  return success();
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLuminousKernelOutliningPass(DispatchBuilderFn fn) {
  return std::make_unique<LuminousKernelOutliningPass>(std::move(fn));
}
