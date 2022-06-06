// RUN: mlir-opt -convert-linalg-to-luminous-launch -allow-unregistered-dialect %s | FileCheck %s

module {
  func.func @test(%arg0: memref<1024x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    linalg.matvec {"linalg-max-memory-footprint" = 10000 : i64} ins(%arg0, %arg1 : memref<1024x1024xf32>, memref<1024xf32>) outs(%arg2 : memref<1024xf32>)
    return
  }
}

// CHECK:       module {
// CHECK:         func.func @test(%[[ARG_0:.*]]: memref<1024x1024xf32>, %[[ARG_1:.*]]: memref<1024xf32>, %[[ARG_2:.*]]: memref<1024xf32>) {
// CHECK:           %[[C1024:.*]] = arith.constant 1024 : index
// CHECK:           luminous.launch shape (%[[C1024]]) step (%[[C1024]]) {
// CHECK:           ^bb0(%[[ARG_3:.*]]):
// CHECK:             linalg.matvec {"linalg-max-memory-footprint" = 10000 : i64} ins(%[[ARG_0]], %[[ARG_1]] : memref<1024x1024xf32>, memref<1024xf32>) outs(%[[ARG_2]] : memref<1024xf32>)
// CHECK:             luminous.yield
// CEHCK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }
