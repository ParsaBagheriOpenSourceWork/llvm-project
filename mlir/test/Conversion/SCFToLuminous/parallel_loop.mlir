// RUN: mlir-opt -convert-parallel-to-luminous-dispatch %s | FileCheck %s

module  {
  func @function() {
    %c1024 = constant 1024 : index
    %c64 = constant 64 : index
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c1024, %c64) step (%c64, %c1) {
      scf.yield
    } {"luminous-launch"}
    return
  }
}

// CHECK:       module {
// CHECK:         func @function() {
// CHECK-DAG:       [[VAL_1024:%.*]] = constant 1024 : index
// CHECK-DAG:       [[VAL_64:%.*]] = constant 64 : index
// CHECK-DAG:       [[VAL_1:%.*]] = constant 1 : index
// CHECK-DAG:       [[VAL_0:%.*]] = constant 0 : index
// CHECK:           luminous.launch shape ([[VAL_1024]], [[VAL_64]]) step ([[VAL_64]], [[VAL_1]]) {
// CHECK:           ^bb0([[ARG_0:%.*]]: index, [[ARG_1:%.*]]: index):  // no predecessors
// CHECK:             luminous.yield
// CEHCK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }
