add_mlir_conversion_library(MLIRLuminousToLLVM
        LuminousToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LuminousToLLVM

    DEPENDS
    MLIRConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRArithmeticDialect
    MLIRControlFlowDialect
    MLIRLuminous
    MLIRLinalgDialect
    MLIRLLVMDialect
    MLIRAsyncDialect
    MLIRSCFDialect
    MLIRTransforms
    )