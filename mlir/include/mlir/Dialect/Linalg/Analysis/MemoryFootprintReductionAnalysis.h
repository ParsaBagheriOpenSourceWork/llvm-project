//===- MemoryFootprintReductionAnalysis.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ANALYSIS_MEMORYFOOTPRINTREDUCTIONANALYSIS_H_
#define MLIR_DIALECT_ANALYSIS_MEMORYFOOTPRINTREDUCTIONANALYSIS_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace linalg {

class LinalgOpShape;

using MemReduceFn = std::function<void(Operation *, LinalgOpShape &, int64_t)>;

/// Encapsulates the information that we need for reducing the memory footprint
/// of a LinalgOp
class LinalgOpShape {
  SmallVector<int64_t, 4> opShape;
  const SmallVector<AffineMap, 4> indexingMaps;
  const SmallVector<int64_t, 4> bitWidths;

  LinalgOpShape(SmallVector<int64_t, 4> s, ArrayRef<AffineMap> iMaps,
                SmallVector<int64_t, 4> bWidths)
      : opShape(std::move(s)), indexingMaps(iMaps.begin(), iMaps.end()),
        bitWidths(std::move(bWidths)) {};

public:
  static LinalgOpShape create(LinalgOp);

  SmallVector<int64_t, 4> &get() { return opShape; }

  /// Computes the size of this linalg operation based on the sizes of its
  /// operands (which are calculated by looking at their corresponding
  /// indexingMap)
  size_t computeSize() const;
};

void reduceLinalgOpFootprintGreedily(Operation *, LinalgOpShape &,
                                     size_t maxSize);

llvm::SmallVector<int64_t, 4> computeTileSizesForMemoryFootprintReduction(
    LinalgOp op, int64_t maxMemoryFootprint,
    MemReduceFn reduceFn = reduceLinalgOpFootprintGreedily);

} // namespace linalg
} // namespace mlir
#endif