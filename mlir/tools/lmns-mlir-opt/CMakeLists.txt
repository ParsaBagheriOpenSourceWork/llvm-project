set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${test_libs}
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  )

add_llvm_tool(lmns-mlir-opt
  lmns-mlir-opt.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(lmns-mlir-opt PRIVATE ${LIBS})
llvm_update_compile_flags(lmns-mlir-opt)

mlir_check_all_link_libraries(lmns-mlir-opt)
