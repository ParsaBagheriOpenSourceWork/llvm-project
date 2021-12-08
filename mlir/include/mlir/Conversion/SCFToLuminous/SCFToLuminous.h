//===- SCFToLuminous.h - Convert loop nests to Luminous kernels -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SCFTOLUMINOUS_SCFTOLUMINOUS_H
#define MLIR_CONVERSION_SCFTOLUMINOUS_SCFTOLUMINOUS_H


#include "mlir/Support/LLVM.h"

namespace mlir {
class AffineForOp;
class ConversionTarget;
struct LogicalResult;
class MLIRContext;
class Value;
class Operation;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

namespace scf {
class ForOp;
} // end namespace scf

/// Adds the conversion pattern from `scf.parallel` to `luminous.dispatch` to
/// the provided pattern list.
void populateParallelLoopToLuminousPatterns(RewritePatternSet &patterns);

/// Configures the rewrite target such that only `scf.parallel` operations that
/// are not rewritten by the provided patterns are legal.
void configureParallelLoopToLuminousLegality(ConversionTarget &target);

/// Clean up after applyPartialConversion/applyFullConversion call.
void finalizeParallelLoopToLuminousConversion(Operation *op);

} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOLUMINOUS_SCFTOLUMINOUS_H
