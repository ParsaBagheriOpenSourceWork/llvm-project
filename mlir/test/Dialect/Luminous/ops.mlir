// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

module attributes {luminous.container_module} {
  // CHECK-LABEL: luminous.module @kernels
  luminous.module @kernels {
    luminous.func @kernel (%arg0: memref<1024xf32>, %arg1: memref<2048xf32>) {
      luminous.return
    }
  }

  func.func @f(%arg0: memref<1024xf32>, %arg1: memref<2048xf32>) {
    %t0 = luminous.dispatch @kernels::@kernel(%arg0: memref<1024xf32>, %arg1: memref<2048xf32>)
    return
  }
}
