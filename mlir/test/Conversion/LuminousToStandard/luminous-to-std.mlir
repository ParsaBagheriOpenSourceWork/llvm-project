// RUN: mlir-opt -convert-luminous-to-std %s | FileCheck %s

module {
  // CHECK: module @m attributes {luminous.module} {
  luminous.module @m{
    // CHECK: func.func @fn(%[[ARG:.*]]: memref<100xf32>) attributes {luminous.func} {
    luminous.func @fn(%arg0: memref<100xf32>) {
      // CHECK: return
      luminous.return
    }
  }
}