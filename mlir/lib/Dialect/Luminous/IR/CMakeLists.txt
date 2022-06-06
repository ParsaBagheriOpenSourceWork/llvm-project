add_mlir_dialect_library(MLIRLuminous
    Luminous.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Luminous

    DEPENDS
    MLIRLuminousOpsIncGen

    LINK_LIBS PUBLIC
    MLIRDialect
    MLIRIR
    )