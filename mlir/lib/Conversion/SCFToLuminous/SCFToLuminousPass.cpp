//
// Created by Parsa Bagheri on 12/7/21.
//

#include "mlir/Conversion/SCFToLuminous/SCFToLuminousPass.h"

using namespace mlir;
using namespace mlir::scf;

namespace {
// A pass that traverses top-level loops in the function and converts them to
// GPU launch operations.  Nested launches are not allowed, so this does not
// walk the function recursively to avoid considering nested loops.
struct ForLoopMapper : public ConvertAffineForToGPUBase<ForLoopMapper> {
  ForLoopMapper() = default;
  ForLoopMapper(unsigned numBlockDims, unsigned numThreadDims) {
    this->numBlockDims = numBlockDims;
    this->numThreadDims = numThreadDims;
  }

  void runOnFunction() override {
    for (Operation &op : llvm::make_early_inc_range(getFunction().getOps())) {
      if (auto forOp = dyn_cast<AffineForOp>(&op)) {
        if (failed(convertAffineLoopNestToGPULaunch(forOp, numBlockDims,
                                                    numThreadDims)))
          signalPassFailure();
      }
    }
  }
};

struct ParallelLoopToGpuPass
    : public ConvertParallelLoopToGpuBase<ParallelLoopToGpuPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateParallelLoopToGPUPatterns(patterns);
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    configureParallelLoopToGPULegality(target);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
    finalizeParallelLoopToGPUConversion(getOperation());
  }
};

} // namespace

std::unique_ptr<Pass> createParallelToLuminousDispatchPass() {

}