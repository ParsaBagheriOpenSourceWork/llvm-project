// RUN: mlir-opt --luminous-kernel-outlining -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0)>
#map = affine_map<(d0) -> (d0)>

// CHECK: module attributes {luminous.container_module}
module {

// CHECK-LABEL: luminous.module @device_module
// CHECK-LABEL: luminous.func @async_fn_0
// CHECK-SAME: (%[[KERNEL_ARG0:.*]]: memref<1024xf32>, %[[KERNEL_ARG1:.*]]: memref<1024xf32>, %[[KERNEL_ARG2:.*]]: memref<1024xf32>)
// CHECK-NEXT: linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[KERNEL_ARG0]], %[[KERNEL_ARG1]] : memref<1024xf32>, memref<1024xf32>) outs(%[[KERNEL_ARG2]] : memref<1024xf32>)
// CHECK: luminous.return

  // CHECK-LABEL: func.func @test1
  // CHECK: (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]])
  func.func @test1(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    // CHECK: %[[C1024:.*]] = arith.constant 1024 : index
    %c1024 = arith.constant 1024 : index
    // CHECK: luminous.launch shape (%[[C1024]]) step (%[[C1024]])
    luminous.launch shape (%c1024) step (%c1024){
    // CHECK: ^[[BLOCK:.*]](%[[ARG3:.*]]):
    ^bb0(%arg3: index):
      // CHECK: %[[DISP:.*]] = luminous.dispatch  @device_module::@async_fn_0 (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]])
      // CHECK: async.await %[[DISP]] : !async.token
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<1024xf32>, memref<1024xf32>) outs(%arg2 : memref<1024xf32>) attrs =  {"linalg-max-memory-footprint" = 12294 : i64} {
      ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
        %0 = arith.addf %arg4, %arg5 : f32
        linalg.yield %0 : f32
      }
      // CHECK: luminous.yield
      luminous.yield
    }
    return
  }
}

// -----
module {
  func.func @test2(%arg0: memref<1024x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>, %arg3: memref<f32>) {
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    luminous.launch shape (%c1024) step (%c1024){
    ^bb0(%arg4: index):
      linalg.matvec {"linalg-max-memory-footprint" = 10000 : i64} ins(%arg0, %arg1 : memref<1024x1024xf32>, memref<1024xf32>) outs(%arg2 : memref<1024xf32>)
      linalg.matvec {"linalg-max-memory-footprint" = 10000 : i64} ins(%arg0, %arg2 : memref<1024x1024xf32>, memref<1024xf32>) outs(%arg1 : memref<1024xf32>)
      linalg.dot {"linalg-max-memory-footprint" = 10000 : i64} ins(%arg1, %arg2 : memref<1024xf32>, memref<1024xf32>) outs(%arg3 : memref<f32>)
      luminous.yield
    }
    return
  }
}
// CHECK-LABEL: luminous.module @device_module
// CHECK-LABEL: luminous.func @async_fn_1
// CHECK-LABEL: luminous.func @async_fn_2
// CHECK-LABEL: luminous.func @async_fn_3
// CHECK-LABEL: func.func @test2
    // CHECK: (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]], %[[ARG3:.*]])
    // CHECK: %[[C1024:.*]] = arith.constant 1024 : index
    // CHECK: luminous.launch shape (%[[C1024]]) step (%[[C1024]])
    // CHECK: ^[[BLOCK:.*]](%[[ARG4:.*]]):
        // CHECK: %[[DISP0:.*]] = luminous.dispatch  @device_module::@async_fn_1 (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]])
        // CHECK: %[[DISP1:.*]] = luminous.dispatch  [%[[DISP0]]] @device_module::@async_fn_2 (%[[ARG0:.*]], %[[ARG2:.*]], %[[ARG1:.*]])
        // CHECK: %[[DISP2:.*]] = luminous.dispatch  [%[[DISP1]]] @device_module::@async_fn_3 (%[[ARG1:.*]], %[[ARG2:.*]], %[[ARG3:.*]])
        // CHECK: async.await %[[DISP2]] : !async.token