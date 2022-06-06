// RUN: mlir-opt -convert-parallel-to-luminous-launch %s | FileCheck %s

module  {
  func.func @function() {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    scf.parallel (%arg0) = (%c0) to (%c64) step (%c1) {
      scf.yield
    } {"luminous-launch"}
    return
  }
}

// CHECK:       module {
// CHECK:         func.func @function() {
// CHECK-DAG:       %[[VAL_64:.*]] = arith.constant 64 : index
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK:           luminous.launch shape (%[[VAL_64]]) step (%[[VAL_1]]) {
// CHECK:           ^bb0(%[[ARG_0:.*]]):
// CHECK:             luminous.yield
// CEHCK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }
