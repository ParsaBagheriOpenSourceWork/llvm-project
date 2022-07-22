// RUN: mlir-opt -convert-luminous-to-llvm %s | FileCheck %s

module attributes {luminous.container_module} {
  module @device_module attributes {luminous.module} {
    llvm.func @async_fn_0(%arg0: i32) attributes {luminous.func} {
      llvm.return
    }
  }
  func.func @fn_0() {
    %c0 = arith.constant 0 : i32
    %1 = luminous.dispatch  @device_module::@async_fn_0 (%c0: i32)
    async.await %1 : !async.token
    return
  }
}

// CHECK: module
// CHECK: llvm.func @__luminous_runtime_dispatch(!llvm.ptr<i8>, !llvm.ptr<i8>, i32)
// CHECK: llvm.func @__luminous_runtime_load_capsule(!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK: llvm.mlir.global internal constant @async_fn_0_wrapper("async_fn_0_wrapper\00")
// CHECK: module @device_module
// CHECK: llvm.func @async_fn_0(%arg0: i32)
// CHECK: llvm.func @async_fn_0_wrapper(%arg0: !llvm.ptr<i8>)
// CHECK-DAG:   %[[CAST:.*]] = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<struct<packed (i32)>>
// CHECK-DAG:   %[[C_0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:   %[[ELM_PTR:.*]] = llvm.getelementptr %[[CAST]][%[[C_0]], 0] : (!llvm.ptr<struct<packed (i32)>>, i32) -> !llvm.ptr<i32>
// CHECK:   %[[ELM:.*]] = llvm.load %[[ELM_PTR]] : !llvm.ptr<i32>
// CHECK:   llvm.call @async_fn_0(%[[ELM]]) : (i32) -> ()
// CHECK:   llvm.return

// CHECK: func.func @fn_0()
// CHECK-DAG:   %[[C0_i32:.*]] = arith.constant 0 : i32
// CHECK-DAG:   %[[C_0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   %[[C_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:   %[[ALLOCA:.*]] = llvm.alloca %[[C_1]] x !llvm.struct<packed (i32)> : (i32) -> !llvm.ptr<struct<packed (i32)>>
// CHECK-DAG:   %[[PTR0:.*]] = llvm.getelementptr %[[ALLOCA]][%[[C_0]], 0] : (!llvm.ptr<struct<packed (i32)>>, i32) -> !llvm.ptr<i32>
// CHECK-DAG:   llvm.store %[[C0_i32]], %[[PTR0]] : !llvm.ptr<i32>
// CHECK-DAG:   %[[PTR1:.*]] = llvm.mlir.addressof @async_fn_0_wrapper : !llvm.ptr<array<19 x i8>>
// CHECK-DAG:   %[[PTR2:.*]] = llvm.getelementptr %[[PTR1]][%[[C_0]], %[[C_0]]] : (!llvm.ptr<array<19 x i8>>, i32, i32) -> !llvm.ptr<i8>
// CHECK-DAG:   %[[FN_PTR:.*]] = llvm.call @__luminous_runtime_load_capsule(%[[PTR2]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK-DAG:   %[[ARGS_PTR:.*]] = llvm.bitcast %[[ALLOCA]] : !llvm.ptr<struct<packed (i32)>> to !llvm.ptr<i8>
// CHECK-DAG:   %[[ARGS_SIZE:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK: llvm.call @__luminous_runtime_dispatch(%[[FN_PTR]], %[[ARGS_PTR]], %[[ARGS_SIZE]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i32) -> ()
// CHECK: return