add_mlir_dialect_library(MLIRLuminousTransforms
    KernelOutlining.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Luminous

    DEPENDS
    MLIRLuminousPassIncGen

    LINK_LIBS PUBLIC
    MLIRAsync
    MLIRLuminous
    MLIRIR
    MLIRMemRef
    MLIRLinalg
    MLIRSCF
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    )
