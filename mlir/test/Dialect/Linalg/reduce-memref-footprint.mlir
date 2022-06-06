// RUN: mlir-opt %s --linalg-memory-footprint-reduce="linalg-max-memory-footprint=5000000" | FileCheck %s -check-prefix=REDUCE-5mb
// RUN: mlir-opt %s --linalg-memory-footprint-reduce="linalg-max-memory-footprint=1000000" | FileCheck %s -check-prefix=REDUCE-1mb
// RUN: mlir-opt %s --linalg-memory-footprint-reduce="linalg-max-memory-footprint=10000" | FileCheck %s -check-prefix=REDUCE-10kb

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @function(%arg0: memref<64x128x1024xf32>, %arg1: memref<64x128x1024xf32>, %arg2: memref<64x128x1024xf32>) {
  linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<64x128x1024xf32>, memref<64x128x1024xf32>) outs(%arg2 : memref<64x128x1024xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %1 = arith.addf %arg3, %arg4 : f32
    linalg.yield %1 : f32
  }
  return
}

// REDUCE-5mb-LABEL: func.func @function(
// REDUCE-5mb-SAME:    [[LHS:%.*]]: {{.*}}, [[RHS:%.*]]: {{.*}}, [[SUM:%.*]]: {{.*}}) {
// REDUCE-5mb-DAG: [[C0:%.*]] = arith.constant 0 : index
// REDUCE-5mb-DAG: [[C64:%.*]] = arith.constant 64 : index
// REDUCE-5mb-DAG: [[C2:%.*]] = arith.constant 2 : index
// REDUCE-5mb: scf.parallel ([[I:%.*]]) = ([[C0]]) to ([[C64]]) step ([[C2]]) {
// REDUCE-5mb-NO: scf.parallel
// REDUCE-5mb:   [[LHS_SUBVIEW:%.*]] = memref.subview [[LHS]]
// REDUCE-5mb:   [[RHS_SUBVIEW:%.*]] = memref.subview [[RHS]]
// REDUCE-5mb:   [[SUM_SUBVIEW:%.*]] = memref.subview [[SUM]]
// REDUCE-5mb:   linalg.generic {{.*}} ins([[LHS_SUBVIEW]], [[RHS_SUBVIEW]]{{.*}} outs([[SUM_SUBVIEW]]


// REDUCE-1mb-LABEL: func.func @function(
// REDUCE-1mb-SAME:    [[LHS:%.*]]: {{.*}}, [[RHS:%.*]]: {{.*}}, [[SUM:%.*]]: {{.*}}) {
// REDUCE-1mb-DAG: [[C0:%.*]] = arith.constant 0 : index
// REDUCE-1mb-DAG: [[C128:%.*]] = arith.constant 128 : index
// REDUCE-1mb-DAG: [[C64:%.*]] = arith.constant 64 : index
// REDUCE-1mb-DAG: [[C1:%.*]] = arith.constant 1 : index
// REDUCE-1mb: scf.parallel ([[I1:%.*]], [[I2:%.*]]) = ([[C0]], [[C0]]) to ([[C64]], [[C128]]) step ([[C1]], [[C64]]) {
// REDUCE-1mb-NO: scf.parallel
// REDUCE-1mb:   [[LHS_SUBVIEW:%.*]] = memref.subview [[LHS]]
// REDUCE-1mb:   [[RHS_SUBVIEW:%.*]] = memref.subview [[RHS]]
// REDUCE-1mb:   [[SUM_SUBVIEW:%.*]] = memref.subview [[SUM]]
// REDUCE-1mb:   linalg.generic {{.*}} ins([[LHS_SUBVIEW]], [[RHS_SUBVIEW]]{{.*}} outs([[SUM_SUBVIEW]]


// REDUCE-10kb-LABEL: func.func @function(
// REDUCE-10kb-SAME:    [[LHS:%.*]]: {{.*}}, [[RHS:%.*]]: {{.*}}, [[SUM:%.*]]: {{.*}}) {
// REDUCE-10kb-DAG: [[C0:%.*]] = arith.constant 0 : index
// REDUCE-10kb-DAG: [[C1024:%.*]] = arith.constant 1024 : index
// REDUCE-10kb-DAG: [[C128:%.*]] = arith.constant 128 : index
// REDUCE-10kb-DAG: [[C64:%.*]] = arith.constant 64 : index
// REDUCE-10kb-DAG: [[C512:%.*]] = arith.constant 512 : index
// REDUCE-10kb-DAG: [[C1:%.*]] = arith.constant 1 : index
// REDUCE-10kb: scf.parallel ([[I1:%.*]], [[I2:%.*]], [[I3:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[C64]], [[C128]], [[C1024]]) step ([[C1]], [[C1]], [[C512]]) {
// REDUCE-10kb-NO: scf.parallel
// REDUCE-10kb:   [[LHS_SUBVIEW:%.*]] = memref.subview [[LHS]]
// REDUCE-10kb:   [[RHS_SUBVIEW:%.*]] = memref.subview [[RHS]]
// REDUCE-10kb:   [[SUM_SUBVIEW:%.*]] = memref.subview [[SUM]]
// REDUCE-10kb:   linalg.generic {{.*}} ins([[LHS_SUBVIEW]], [[RHS_SUBVIEW]]{{.*}} outs([[SUM_SUBVIEW]]

#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
func.func @function_bcast(%arg0: memref<64x128x1024xf32>, %arg1: memref<128x1024xf32>, %arg2: memref<64x128x1024xf32>) {
  linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<64x128x1024xf32>, memref<128x1024xf32>) outs(%arg2 : memref<64x128x1024xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %1 = arith.addf %arg3, %arg4 : f32
    linalg.yield %1 : f32
  }
  return
}


// REDUCE-5mb-LABEL: func.func @function_bcast(
// REDUCE-5mb-SAME:    [[LHS:%.*]]: {{.*}}, [[RHS:%.*]]: {{.*}}, [[SUM:%.*]]: {{.*}}) {
// REDUCE-5mb-DAG: [[C0:%.*]] = arith.constant 0 : index
// REDUCE-5mb-DAG: [[C64:%.*]] = arith.constant 64 : index
// REDUCE-5mb-DAG: [[C4:%.*]] = arith.constant 4 : index
// REDUCE-5mb: scf.parallel ([[I:%.*]]) = ([[C0]]) to ([[C64]]) step ([[C4]]) {
// REDUCE-5mb-NO: scf.parallel
// REDUCE-5mb:   [[LHS_SUBVIEW:%.*]] = memref.subview [[LHS]]
// REDUCE-5mb:   [[SUM_SUBVIEW:%.*]] = memref.subview [[SUM]]
// REDUCE-5mb:   linalg.generic {{.*}} ins([[LHS_SUBVIEW]], [[RHS]]{{.*}} outs([[SUM_SUBVIEW]]


// REDUCE-1mb-LABEL: func.func @function_bcast(
// REDUCE-1mb-SAME:    [[LHS:%.*]]: {{.*}}, [[RHS:%.*]]: {{.*}}, [[SUM:%.*]]: {{.*}}) {
// REDUCE-1mb-DAG: [[C0:%.*]] = arith.constant 0 : index
// REDUCE-1mb-DAG: [[C128:%.*]] = arith.constant 128 : index
// REDUCE-1mb-DAG: [[C64:%.*]] = arith.constant 64 : index
// REDUCE-1mb-DAG: [[C1:%.*]] = arith.constant 1 : index
// REDUCE-1mb: scf.parallel ([[I1:%.*]], [[I2:%.*]]) = ([[C0]], [[C0]]) to ([[C64]], [[C128]]) step ([[C1]], [[C64]]) {
// REDUCE-1mb-NO: scf.parallel
// REDUCE-1mb:   [[LHS_SUBVIEW:%.*]] = memref.subview [[LHS]]
// REDUCE-1mb:   [[RHS_SUBVIEW:%.*]] = memref.subview [[RHS]]
// REDUCE-1mb:   [[SUM_SUBVIEW:%.*]] = memref.subview [[SUM]]
// REDUCE-1mb:   linalg.generic {{.*}} ins([[LHS_SUBVIEW]], [[RHS_SUBVIEW]]{{.*}} outs([[SUM_SUBVIEW]]

// REDUCE-10kb-LABEL: func.func @function_bcast(
// REDUCE-10kb-SAME:    [[LHS:%.*]]: {{.*}}, [[RHS:%.*]]: {{.*}}, [[SUM:%.*]]: {{.*}}) {
// REDUCE-10kb-DAG: [[C0:%.*]] = arith.constant 0 : index
// REDUCE-10kb-DAG: [[C128:%.*]] = arith.constant 128 : index
// REDUCE-10kb-DAG: [[C64:%.*]] = arith.constant 64 : index
// REDUCE-10kb-DAG: [[C1:%.*]] = arith.constant 1 : index
// REDUCE-10kb-DAG: [[C512:%.*]] = arith.constant 512 : index
// REDUCE-10kb-DAG: [[C1024:%.*]] = arith.constant 1024 : index
// REDUCE-10kb: scf.parallel ([[I1:%.*]], [[I2:%.*]], [[I3:%.*]]) = ([[C0]], [[C0]], [[C0]]) to ([[C64]], [[C128]], [[C1024]]) step ([[C1]], [[C1]], [[C512]]) {
// REDUCE-10kb-NO: scf.parallel
// REDUCE-10kb:   [[LHS_SUBVIEW:%.*]] = memref.subview [[LHS]]
// REDUCE-10kb:   [[RHS_SUBVIEW:%.*]] = memref.subview [[RHS]]
// REDUCE-10kb:   [[SUM_SUBVIEW:%.*]] = memref.subview [[SUM]]
// REDUCE-10kb:   linalg.generic {{.*}} ins([[LHS_SUBVIEW]], [[RHS_SUBVIEW]]{{.*}} outs([[SUM_SUBVIEW]]
