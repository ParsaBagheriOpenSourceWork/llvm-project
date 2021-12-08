// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

module attributes {luminous.container_module} {
  luminous.module @kernels {
    luminous.func @kernel_1(%arg0: memref<?xf32, 1>) {
      luminous.return
    }
    luminous.func @kernel_2(%arg0: f32, %arg1: memref<?xf32, 1>) {
      luminous.return
    }
  }

  func @function(%arg0 : f32, %arg1 : memref<?xf32, 1>) {
    %c64 = constant 64 : index
    %c128 = constant 128 : index
    %t0 = luminous.dispatch @kernels::@kernel_1 <%c64,%c128> (%arg1 : memref<?xf32, 1>)
    %t1 = luminous.dispatch [%t0] @kernels::@kernel_2 <%c64,%c128> (%arg0 : f32, %arg1 : memref<?xf32, 1>)
    luminous.wait [%t1]
    return
  }
}
