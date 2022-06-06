//===- MemoryFootprintReductionAnalysis.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Analysis/MemoryFootprintReductionAnalysis.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>

using namespace mlir;
using namespace mlir::linalg;

static int64_t getNextDivisor(int64_t n, int64_t divisor) {
  assert(divisor <= n && divisor >= 1);
  while (n % divisor)
    divisor++;
  return divisor;
}

/// Calculates the size (in bytes) of a ranked tensor
static size_t getSizeFromShape(llvm::ArrayRef<int64_t> shape,
                               size_t elementBitWidth) {
  assert(elementBitWidth % 8 == 0 && "BitWidth has to be divisible by 8");
  auto elementSize = elementBitWidth / 8;
  return std::accumulate(shape.begin(), shape.end(), elementSize,
                         std::multiplies<>());
}

static SmallVector<int64_t, 4> getBitWidths(LinalgOp op) {
  SmallVector<int64_t, 4> bitWidths;
  for (auto operand : op->getOperands()) {
    if (auto shapedType = operand.getType().dyn_cast<ShapedType>()) {
      bitWidths.push_back(shapedType.getElementTypeBitWidth());
    } else {
      // TODO: Find a method for calculating the bitwidth of the type when
      // it's not shaped
      bitWidths.push_back(0);
    }
  }
  return bitWidths;
}

namespace mlir {
namespace linalg {

LinalgOpShape LinalgOpShape::create(LinalgOp op) {
  auto shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  auto opShape = shapeSizesToLoopsMap.compose(op.getStaticShape());
  auto indexingMaps = op.getIndexingMaps();
  auto bitWidths = getBitWidths(op);
  assert(bitWidths.size() == indexingMaps.size());
  return LinalgOpShape(opShape, indexingMaps, std::move(bitWidths));
}

size_t LinalgOpShape::computeSize() const {
  return std::accumulate(
      zip(indexingMaps, bitWidths).begin(), zip(indexingMaps, bitWidths).end(),
      0, [this](size_t curSize, const std::tuple<AffineMap, int64_t> &it) {
        auto &map = std::get<0>(it);
        auto bitWidth = std::get<1>(it);
        auto curOperandsShape = map.compose(opShape);
        return curSize + getSizeFromShape(curOperandsShape, bitWidth);
      });
}

void reduceLinalgOpFootprintGreedily(Operation *op,
                                     LinalgOpShape &linalgOpShape,
                                     size_t maxSize) {
  // Filtering out unwanted ops
  if (!isa<LinalgOp>(op))
    return;

  auto &shape = linalgOpShape.get();
  for (size_t i = 0, end = shape.size(); i < end; i++) {
    auto curSize = linalgOpShape.computeSize();
    if (curSize <= maxSize)
      return;
    int64_t reductionFactor = (curSize + maxSize - 1) / maxSize;
    if (shape[i] < reductionFactor)
      shape[i] = 1;
    else {
      auto divisor = getNextDivisor(shape[i], reductionFactor);
      shape[i] /= divisor;
    }
  }
}

SmallVector<int64_t, 4> computeTileSizesForMemoryFootprintReduction(
    LinalgOp op, int64_t maxMemoryFootprint, MemReduceFn reduceFn) {
  auto opShape = LinalgOpShape::create(op);
  reduceFn(op, opShape, maxMemoryFootprint);
  assert(opShape.computeSize() <= maxMemoryFootprint &&
         "memory reduction function didn't reduce memory footprint of op to a "
         "point below the maximum bytes allowed.");
  return opShape.get();
}

} // namespace linalg
} // namespace mlir