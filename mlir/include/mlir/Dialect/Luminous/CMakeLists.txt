add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Luminous)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Luminous)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Luminous)
add_public_tablegen_target(MLIRLuminousPassIncGen)

add_mlir_doc(Passes LuminousPasses ./ -gen-pass-doc)
