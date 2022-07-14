//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugCounter.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

struct ReplaceFuncWithDecl : public OpRewritePattern<ModuleOp> {
  using OpRewritePattern<ModuleOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ModuleOp op,
                                PatternRewriter &rewriter) const override {

//    auto parentModule = op->getParentOfType<ModuleOp>();
//    OpBuilder::InsertionGuard g(rewriter);
//    rewriter.setInsertionPoint(op);
//    for (auto &bodyOp : *op.getBody()) {
//      if (auto fnOp = dyn_cast<LLVM::LLVMFuncOp>(bodyOp)) {
//        auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(
//            fnOp->getLoc(), fnOp.getName(), fnOp.getFunctionType());
//        funcOp->setAttr("llvm.emit_c_interface",
//                        UnitAttr::get(op->getContext()));
//        funcOp.setPrivate();
//      }
//    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ModuleExtractor
    : public PassWrapper<ModuleExtractor, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<ReplaceFuncWithDecl>(&getContext());
    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
  }
};

static llvm::Optional<OwningOpRef<ModuleOp>>
readModule(const char *mlirFile, mlir::MLIRContext *mlirContext) {
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, mlirContext);
  auto OutputBuf = llvm::MemoryBuffer::getFile(mlirFile);
  if (!OutputBuf)
    return llvm::None;
  sourceMgr.AddNewSourceBuffer(std::move(OutputBuf.get()), llvm::SMLoc());
  auto theModule = parseSourceFile<ModuleOp>(sourceMgr, mlirContext);
  if (!theModule)
    return llvm::None;
  return mlir::OwningOpRef<mlir::ModuleOp>(std::move(theModule));
}

static LogicalResult extractModuleAndAddFuncDecls(ModuleOp module) {
  auto *mlirContext = module.getContext();
  PassManager pm(mlirContext, OpPassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<ModuleExtractor>());
  return pm.run(module);
}

static LogicalResult lmnsMlirOptMain(int argc, char **argv,
                                     llvm::StringRef toolName,
                                     DialectRegistry &registry,
                                     bool preloadDialectsInContext) {
  static cl::opt<std::string> inputFilename(
      cl::Positional, cl::desc("<input file>"), cl::init("-"));

  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));

  static cl::opt<bool> splitInputFile(
      "split-input-file",
      cl::desc("Split the input file into pieces and process each "
               "chunk independently"),
      cl::init(false));

  static cl::opt<bool> verifyDiagnostics(
      "verify-diagnostics",
      cl::desc("Check that emitted diagnostics match "
               "expected-* lines on the corresponding line"),
      cl::init(false));

  static cl::opt<bool> verifyPasses(
      "verify-each",
      cl::desc("Run the verifier after each transformation pass"),
      cl::init(true));

  static cl::opt<bool> allowUnregisteredDialects(
      "allow-unregistered-dialect",
      cl::desc("Allow operation with no registered dialects"), cl::init(false));

  InitLLVM y(argc, argv);

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  DebugCounter::registerCLOptions();
  PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Build the list of dialects as a header for the --help message.
  std::string helpHeader = (toolName + "\nAvailable Dialects: ").str();
  {
    llvm::raw_string_ostream os(helpHeader);
    interleaveComma(registry.getDialectNames(), os,
                    [&](auto name) { os << name; });
  }
  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, helpHeader);

  mlir::MLIRContext mlirContext(registry, MLIRContext::Threading::DISABLED);
  mlirContext.allowUnregisteredDialects();
  auto module = readModule(inputFilename.c_str(), &mlirContext);
  if (!module) {
    llvm::errs() << "failed to parse module\n";
    return failure();
  }

  int tmpFileFD;
  SmallString<256> tmpFilePath;
  if (auto ec =
          sys::fs::createTemporaryFile("", "mlir", tmpFileFD, tmpFilePath)) {
    llvm::errs() << "error: " << toString(llvm::errorCodeToError(ec)) << "\n";
    return failure();
  }
  llvm::raw_fd_ostream tmpOut(tmpFileFD, true);
  auto moduleOp = module->get();
  for (auto &op : *moduleOp.getBody()) {
    if (isa<ModuleOp>(op)) {
      op.print(tmpOut);
    }
  }
  tmpOut.close();

  if (failed(extractModuleAndAddFuncDecls(moduleOp))) {
    llvm::errs() << "failed to extract inner module\n";
    return failure();
  }

  // the inner module is now taken out of the moduleOp and is written to tempFile
  // dump moduleOp it to stderr for now TODO set up cl arg
  llvm::errs() << moduleOp;

  // run mlir opt on the inner module now
  std::string errorMessage;
  auto file = openInputFile(tmpFilePath, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  if (failed(MlirOptMain(output->os(), std::move(file), passPipeline, registry,
                         splitInputFile, verifyDiagnostics, verifyPasses,
                         allowUnregisteredDialects, preloadDialectsInContext)))
    return failure();

  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return success();
}

int main(int argc, char **argv) {
  registerAllPasses();
  DialectRegistry registry;
  registerAllDialects(registry);
  return mlir::asMainReturnCode(
      lmnsMlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry,
                      /*preloadDialectsInContext=*/false));
}
