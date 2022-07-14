//
// Created by parsab on 6/26/22.
//

#ifndef MLIR_CONVERSION_LUMINOUSTOLLVM_LUMINOUSTOLLVM_H
#define MLIR_CONVERSION_LUMINOUSTOLLVM_LUMINOUSTOLLVM_H

#include <memory>

namespace mlir {
template <typename T>
class OperationPass;
class ModuleOp;
std::unique_ptr<OperationPass<ModuleOp>>
createConvertLuminousToStandardPass();
} // namespace mlir


#endif // MLIR_CONVERSION_LUMINOUSTOLLVM_LUMINOUSTOLLVM_H