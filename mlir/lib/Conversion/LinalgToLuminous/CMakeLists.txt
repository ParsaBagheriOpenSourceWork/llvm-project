add_mlir_conversion_library(MLIRLinalgToLuminous
    LinalgToLuminous.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LinalgToLuminous

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