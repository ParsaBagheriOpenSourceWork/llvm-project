//
// Created by parsab on 7/14/22.
//

#ifndef LLVM_LUMINOUSTOLLVM_H
#define LLVM_LUMINOUSTOLLVM_H

#include <memory>

namespace mlir {
template <typename T>
class OperationPass;
class ModuleOp;
std::unique_ptr<OperationPass<ModuleOp>>
createConvertLuminousToLLVMPass();
} // namespace mlir


#endif // LLVM_LUMINOUSTOLLVM_H
