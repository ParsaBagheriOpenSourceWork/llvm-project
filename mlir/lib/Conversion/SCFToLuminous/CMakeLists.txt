add_mlir_conversion_library(MLIRSCFToLuminous
    SCFToLuminous.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToLuminous

    DEPENDS
    MLIRConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRArithmetic
    MLIRControlFlow
    MLIRLuminous
    MLIRAsync
    MLIRSCF
    MLIRTransforms
    )