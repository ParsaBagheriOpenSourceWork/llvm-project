add_mlir_conversion_library(MLIRLuminousToStandard
    LuminousToStandard.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LuminousToStandard

    DEPENDS
    MLIRConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRArithmetic
    MLIRControlFlow
    MLIRLuminous
    MLIRLinalg
    MLIRAsync
    MLIRSCF
    MLIRTransforms
    )