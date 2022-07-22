// RUN: mlir-opt --luminous-kernel-outlining -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0)>
#map = affine_map<(d0) -> (d0)>

// CHECK: module attributes {luminous.container_module}
module {

// CHECK: luminous.module @[[CAPSULE:.*]]
// CHECK-NEXT: luminous.func @[[ASYNC_FN_0:.*]](%[[KERNEL_ARG0:.*]]: memref<1024xf32>, %[[KERNEL_ARG1:.*]]: memref<1024xf32>, %[[KERNEL_ARG2:.*]]: memref<1024xf32>)
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
      // CHECK: %[[DISP:.*]] = luminous.dispatch  @[[CAPSULE:.*]]::@[[ASYNC_FN_0:.*]] (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]])
      // CHECK: async.await %[[DISP]] : !async.token
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<1024xf32>, memref<1024xf32>) outs(%arg2 : memref<1024xf32>) attrs =  {"luminous.memory-footprint" = 12294 : i64} {
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

/// Chaining dispatches

module {
  func.func @test2(%arg0: memref<1024x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>, %arg3: memref<f32>) {
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    luminous.launch shape (%c1024) step (%c1024){
    ^bb0(%arg4: index):
      linalg.matvec {"luminous.memory-footprint" = 10000 : i64} ins(%arg0, %arg1 : memref<1024x1024xf32>, memref<1024xf32>) outs(%arg2 : memref<1024xf32>)
      linalg.matvec {"luminous.memory-footprint" = 10000 : i64} ins(%arg0, %arg2 : memref<1024x1024xf32>, memref<1024xf32>) outs(%arg1 : memref<1024xf32>)
      linalg.dot {"luminous.memory-footprint" = 10000 : i64} ins(%arg1, %arg2 : memref<1024xf32>, memref<1024xf32>) outs(%arg3 : memref<f32>)
      luminous.yield
    }
    return
  }
}

// CHECK: luminous.module @[[CAPSULE:.*]]
// CHECK: luminous.func @[[ASYNC_FN_0:.*]]
// CHECK: luminous.func @[[ASYNC_FN_1:.*]]
// CHECK: luminous.func @[[ASYNC_FN_2:.*]]
// CHECK-LABEL: func.func @test2
    // CHECK: (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]], %[[ARG3:.*]])
    // CHECK: %[[C1024:.*]] = arith.constant 1024 : index
    // CHECK: luminous.launch shape (%[[C1024]]) step (%[[C1024]])
    // CHECK: ^[[BLOCK:.*]](%[[ARG4:.*]]):
        // CHECK: %[[DISP0:.*]] = luminous.dispatch  @[[CAPSULE:.*]]::@[[ASYNC_FN_0:.*]] (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]])
        // CHECK: %[[DISP1:.*]] = luminous.dispatch  [%[[DISP0]]] @[[CAPSULE:.*]]::@[[ASYNC_FN_1:.*]] (%[[ARG0:.*]], %[[ARG2:.*]], %[[ARG1:.*]])
        // CHECK: %[[DISP2:.*]] = luminous.dispatch  [%[[DISP1]]] @[[CAPSULE:.*]]::@[[ASYNC_FN_2:.*]] (%[[ARG1:.*]], %[[ARG2:.*]], %[[ARG3:.*]])
        // CHECK: async.await %[[DISP2]] : !async.token


// -----

/// Dispatch inside for loop
module {
  func.func @test3(%arg0: memref<1024x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    luminous.launch shape (%c1024) step (%c1024){
    ^bb0(%arg4: index):
      scf.for %arg5 = %c0 to %c1024 step %c1024 {
        linalg.matvec {"luminous.memory-footprint" = 10000 : i64} ins(%arg0, %arg1 : memref<1024x1024xf32>, memref<1024xf32>) outs(%arg2 : memref<1024xf32>)
      }
      luminous.yield
    }
    return
  }
}

// CHECK: luminous.module @[[CAPSULE:.*]]
// CHECK-NEXT: luminous.func @[[ASYNC_FN_0:.*]](%[[KERNEL_ARG0:.*]]: memref<1024x1024xf32>, %[[KERNEL_ARG1:.*]]: memref<1024xf32>, %[[KERNEL_ARG2:.*]]: memref<1024xf32>)
// CHECK-NEXT: linalg.matvec {"luminous.memory-footprint" = 10000 : i64} ins(%[[KERNEL_ARG0]], %[[KERNEL_ARG1]] : memref<1024x1024xf32>, memref<1024xf32>) outs(%[[KERNEL_ARG2]] : memref<1024xf32>)
// CHECK-LABEL: func.func @test3
// CHECK: (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]])
    // CHECK: %[[C1024:.*]] = arith.constant 1024 : index
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: luminous.launch shape (%[[C1024]]) step (%[[C1024]])
    // CHECK: ^[[BLOCK:.*]](%[[ARG3:.*]]):
        // CHECK: scf.for %[[ARG4:.*]] = %[[C0:.*]] to %[[C1024:.*]] step %[[C1024:.*]] {
            // CHECK-NEXT: %[[DISP0:.*]] = luminous.dispatch  @[[CAPSULE:.*]]::@[[ASYNC_FN_0:.*]] (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]])
            // CHECK-NEXT: async.await %[[DISP0]] : !async.token


// -----

/// Dispatch inside nested for loops

module {
  func.func @test3(%arg0: memref<1024x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    luminous.launch shape (%c1024) step (%c1024){
    ^bb0(%arg4: index):
      scf.for %arg5 = %c0 to %c1024 step %c1024 {
        scf.for %arg6 = %c0 to %c1024 step %c1024 {
            linalg.matvec {"luminous.memory-footprint" = 10000 : i64} ins(%arg0, %arg1 : memref<1024x1024xf32>, memref<1024xf32>) outs(%arg2 : memref<1024xf32>)
        }
      }
      luminous.yield
    }
    return
  }
}

// CHECK: luminous.module @[[CAPSULE:.*]]
// CHECK-NEXT: luminous.func @[[ASYNC_FN_0:.*]](%[[KERNEL_ARG0:.*]]: memref<1024x1024xf32>, %[[KERNEL_ARG1:.*]]: memref<1024xf32>, %[[KERNEL_ARG2:.*]]: memref<1024xf32>)
// CHECK-NEXT: linalg.matvec {"luminous.memory-footprint" = 10000 : i64} ins(%[[KERNEL_ARG0]], %[[KERNEL_ARG1]] : memref<1024x1024xf32>, memref<1024xf32>) outs(%[[KERNEL_ARG2]] : memref<1024xf32>)
// CHECK-LABEL: func.func @test3
// CHECK: (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]])
    // CHECK: %[[C1024:.*]] = arith.constant 1024 : index
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: luminous.launch shape (%[[C1024]]) step (%[[C1024]])
    // CHECK: ^[[BLOCK:.*]](%[[ARG3:.*]]):
        // CHECK: scf.for %[[ARG4:.*]] = %[[C0:.*]] to %[[C1024:.*]] step %[[C1024:.*]] {
            // CHECK: scf.for %[[ARG5:.*]] = %[[C0:.*]] to %[[C1024:.*]] step %[[C1024:.*]] {
                // CHECK-NEXT: %[[DISP0:.*]] = luminous.dispatch  @[[CAPSULE:.*]]::@[[ASYNC_FN_0:.*]] (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]])
                // CHECK-NEXT: async.await %[[DISP0]] : !async.token