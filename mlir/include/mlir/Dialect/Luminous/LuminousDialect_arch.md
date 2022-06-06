## I. Memory Footprint Reduction Pass

### Synopsis

```
mlir-opt --linalg-memory-footprint-reduce="linalg-max-memory-footprint=<bytes>"
```

This pass will tile all of the linalg ops in an mlir module, based on an optional tile size computation function.
Note that the linalg ops must be bufferized before this pass, otherwise tiling will fail.

### Example

Simple elementwise add of two tensors

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 808 : i32}}  {
  func.func @main(%arg0: tensor<64x128x1024xf32>, %arg1: tensor<64x128x1024xf32>) -> tensor<64x128x1024xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "args_0,args_0_1", outputs = "Identity"}} {
    %0 = linalg.init_tensor [64, 128, 1024] : tensor<64x128x1024xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<64x128x1024xf32>, tensor<64x128x1024xf32>) outs(%0 : tensor<64x128x1024xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %2 = arith.addf %arg2, %arg3 : f32
      linalg.yield %2 : f32
    } -> tensor<64x128x1024xf32>
    return %1 : tensor<64x128x1024xf32>
  }
}
```

After bufferization of linalg (`--linalg-bufferize`)

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 808 : i32}} {
  func.func @main(%arg0: tensor<64x128x1024xf32>, %arg1: tensor<64x128x1024xf32>) -> tensor<64x128x1024xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "args_0,args_0_1", outputs = "Identity"}} {
    %0 = bufferization.to_memref %arg1 : memref<64x128x1024xf32>
    %1 = bufferization.to_memref %arg0 : memref<64x128x1024xf32>
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %2 = memref.alloc() {alignment = 128 : i64} : memref<64x128x1024xf32>
    %3 = memref.alloc() {alignment = 128 : i64} : memref<64x128x1024xf32>
    %4 = bufferization.to_tensor %2 : memref<64x128x1024xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %0 : memref<64x128x1024xf32>, memref<64x128x1024xf32>) outs(%3 : memref<64x128x1024xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %6 = arith.addf %arg2, %arg3 : f32
      linalg.yield %6 : f32
    }
    %5 = bufferization.to_tensor %3 : memref<64x128x1024xf32>
    return %5 : tensor<64x128x1024xf32>
  }
}
```

After memory footprint reduction

```mlir
#map0 = affine_map<(d0, d1, d2)[s0] -> (d0 * 131072 + s0 + d1 * 1024 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 808 : i32}} {
  func.func @main(%arg0: tensor<64x128x1024xf32>, %arg1: tensor<64x128x1024xf32>) -> tensor<64x128x1024xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "args_0,args_0_1", outputs = "Identity"}} {
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_memref %arg1 : memref<64x128x1024xf32>
    %1 = bufferization.to_memref %arg0 : memref<64x128x1024xf32>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<64x128x1024xf32>
    scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c64, %c128, %c1024) step (%c1, %c1, %c512) {
      %4 = memref.subview %1[%arg2, %arg3, %arg4] [1, 1, 512] [1, 1, 1] : memref<64x128x1024xf32> to memref<1x1x512xf32, #map0>
      %5 = memref.subview %0[%arg2, %arg3, %arg4] [1, 1, 512] [1, 1, 1] : memref<64x128x1024xf32> to memref<1x1x512xf32, #map0>
      %6 = memref.subview %2[%arg2, %arg3, %arg4] [1, 1, 512] [1, 1, 1] : memref<64x128x1024xf32> to memref<1x1x512xf32, #map0>
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %5 : memref<1x1x512xf32, #map0>, memref<1x1x512xf32, #map0>) outs(%6 : memref<1x1x512xf32, #map0>) attrs =  {"linalg-max-memory-footprint" = 10000 : i64} {
      ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):
        %7 = arith.addf %arg5, %arg6 : f32
        linalg.yield %7 : f32
      }
      scf.yield
    } {"luminous-launch"}
    %3 = bufferization.to_tensor %2 : memref<64x128x1024xf32>
    return %3 : tensor<64x128x1024xf32>
  }
}
```

### Relevant classes and functions

```c++
struct LinalgMemoryFootprintReductionPass;

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgMemoryFootprintReductionPass(int64_t, MemReduceFn);
```

```c++
class LinalgOpShape {
  SmallVector<int64_t, 4> opShape;
  const SmallVector<AffineMap, 4> indexingMaps;
  const SmallVector<int64_t, 4> bitWidths;
  ...
public:
  static LinalgOpShape create(LinalgOp);
  SmallVector<int64_t, 4> &get();
  size_t computeSize() const;
};
```

Encapsulates the information that we need for tiling a linalg op in order to reduce its memory footprint. `get()`
returns a reference to current shape of the linalg op, the user adjusts the shape
and computes the memory footprint using `computeSize()`.

```c++
using MemReduceFn = std::function<void(Operation *, LinalgOpShape &, int64_t)>;
```

```c++
void reduceLinalgOpFootprintGreedily(Operation *op,
                                     LinalgOpShape &linalgOpShape,
                                     size_t maxSize) {
  // Filtering out unwanted ops
  if (!isa<LinalgOp>(op))
    return;

  auto &shape = linalgOpShape.get();
  for (size_t i = 0, end = shape.size(); i < end; i++) {
    auto curSize = linalgOpShape.computeSize();
    if (curSize <= maxSize)
      return;
    int64_t reductionFactor = (curSize + maxSize - 1) / maxSize;
    if (shape[i] < reductionFactor)
      shape[i] = 1;
    else {
      auto divisor = getNextDivisor(shape[i], reductionFactor);
      shape[i] /= divisor;
    }
  }
}
```

`MemReduceFn` is a user defined function that is called within the pass in order to compute the tile sizes.
It takes the linalg op (Operation *), and the shape of the op wrapped in a LinalgOpShape object, and the maximum memory
footprint.
If the user doesn't provide this function `reduceLinalgOpFootprintGreedily()` is used by default.

## II. Dialect Conversion 
### II.a. Parallel to Luminous Launch Pass (Subject to change)

### Synopsis

```
mlir-opt --convert-parallel-to-luminous-launch
```

This is a lowering (conversion) pass from `scf` dialect to `luminous`.
This pass looks for `scf.parallel` ops with `luimous-launch` attribute and converts them to `luminous.launch`.
Luminous launch keeps the shape of the parallel op, and the strides, and aside from the addition of a
terminator(`luminous.yield`), the body of the parallel op stays unchanged.

### Example

Lowering the example above, (after tiling)

```mlir
#map0 = affine_map<(d0, d1, d2)[s0] -> (d0 * 131072 + s0 + d1 * 1024 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 808 : i32}} {
  func.func @main(%arg0: tensor<64x128x1024xf32>, %arg1: tensor<64x128x1024xf32>) -> tensor<64x128x1024xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "args_0,args_0_1", outputs = "Identity"}} {
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_memref %arg1 : memref<64x128x1024xf32>
    %1 = bufferization.to_memref %arg0 : memref<64x128x1024xf32>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<64x128x1024xf32>
    luminous.launch shape (%c64, %c128, %c1024) step (%c1, %c1, %c512){
    ^bb0(%arg2: index, %arg3: index, %arg4: index):
      %4 = memref.subview %1[%arg2, %arg3, %arg4] [1, 1, 512] [1, 1, 1] : memref<64x128x1024xf32> to memref<1x1x512xf32, #map0>
      %5 = memref.subview %0[%arg2, %arg3, %arg4] [1, 1, 512] [1, 1, 1] : memref<64x128x1024xf32> to memref<1x1x512xf32, #map0>
      %6 = memref.subview %2[%arg2, %arg3, %arg4] [1, 1, 512] [1, 1, 1] : memref<64x128x1024xf32> to memref<1x1x512xf32, #map0>
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %5 : memref<1x1x512xf32, #map0>, memref<1x1x512xf32, #map0>) outs(%6 : memref<1x1x512xf32, #map0>) attrs =  {"linalg-max-memory-footprint" = 10000 : i64} {
      ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):
        %7 = arith.addf %arg5, %arg6 : f32
        linalg.yield %7 : f32
      }
      luminous.yield
    }
    %3 = bufferization.to_tensor %2 : memref<64x128x1024xf32>
    return %3 : tensor<64x128x1024xf32>
  }
}
```

There aren't any user definable functions or callbacks for this pass.

### II.b. Linalg to Luminous Launch Pass (Subject to change)

### Synopsis

```
mlir-opt --convert-linalg-to-luminous-launch
```

This is a conversion pass from `linalg` dialect to `luminous`.
This pass looks for `linalg` ops with `max-memory-footprint` attribute that are not already in launch capsules, and 
creates a `luminous.launch` capsule for them.

### Example

Before
```mlir
module {
  func.func @test(%arg0: memref<1024x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    linalg.matvec {"linalg-max-memory-footprint" = 10000 : i64} ins(%arg0, %arg1 : memref<1024x1024xf32>, memref<1024xf32>) outs(%arg2 : memref<1024xf32>)
    return
  }
}
```
After
```mlir
module {
  func.func @test(%arg0: memref<1024x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    %c1024 = arith.constant 1024 : index
    luminous.launch shape (%c1024) step (%c1024) {
    ^bb0(%arg3: index):
      linalg.matvec {"linalg-max-memory-footprint" = 10000 : i64} ins(%arg0, %arg1 : memref<1024x1024xf32>, memref<1024xf32>) outs(%arg2 : memref<1024xf32>)
      luminous.yield
    }
    return
  }
}
```
## III. Luminous Kernel Outlining (Subject to change)

### Synopsis

```
mlir-opt --luminous-kernel-outlining
```

This pass creates a luminous module and creates functions (kernels) within that module, and inserts dispatch calls to
those kernels.
Determining what ops to dispatch (dispatch blocks) is done by a user defined function. Then the pass will take the
dispatch blocks and outlines their corresponding kernels, then removes the ops and inserts an async dispatch call to their
kernel. Then inserts an await call on the future returned by the async dispatch call.

### Example
```mlir
module {
  func.func @main(%arg0: memref<1024x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>, %arg3: memref<f32>) {
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
```
After outlining
```mlir
module attributes {luminous.container_module} {
  luminous.module @device_module{
    luminous.func @async_fn_0(%arg0: memref<1024x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>){
      linalg.matvec {"linalg-max-memory-footprint" = 10000 : i64} ins(%arg0, %arg1 : memref<1024x1024xf32>, memref<1024xf32>) outs(%arg2 : memref<1024xf32>)
      luminous.return
    }
    luminous.func @async_fn_1(%arg0: memref<1024x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>){
      linalg.matvec {"linalg-max-memory-footprint" = 10000 : i64} ins(%arg0, %arg1 : memref<1024x1024xf32>, memref<1024xf32>) outs(%arg2 : memref<1024xf32>)
      luminous.return
    }
    luminous.func @async_fn_2(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<f32>){
      linalg.dot {"linalg-max-memory-footprint" = 10000 : i64} ins(%arg0, %arg1 : memref<1024xf32>, memref<1024xf32>) outs(%arg2 : memref<f32>)
      luminous.return
    }
  }
  func.func @main(%arg0: memref<1024x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>, %arg3: memref<f32>) {
    %c1024 = arith.constant 1024 : index
    luminous.launch shape (%c1024) step (%c1024){
    ^bb0(%arg4: index):
      %0 = luminous.dispatch  @device_module::@async_fn_0 (%arg0: memref<1024x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>)
      %1 = luminous.dispatch [%0] @device_module::@async_fn_1 (%arg0: memref<1024x1024xf32>, %arg2: memref<1024xf32>, %arg1: memref<1024xf32>)
      %2 = luminous.dispatch [%1] @device_module::@async_fn_2 (%arg1: memref<1024xf32>, %arg2: memref<1024xf32>, %arg3: memref<f32>)
      async.await %2 : !async.token
      luminous.yield
    } {kernels_outlined}
    return
  }
}
```

### Relevant classes and functions

```c++
struct LuminousKernelOutliningPass;

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLuminousKernelOutliningPass(DispatchBuilderFn);
```

```c++
class DispatchBlock {
  ...
public:
  void pushBack(Operation *op);
};
```

```c++
class DispatchBlocks {
  ...
public:
  DispatchBlock addNewBlock(llvm::ArrayRef<DispatchBlock> dependencies = {},
                            const std::string &name = "");
};
```

`DispatchBlock` is a user modifiable object. A dispatch block consist of one or more ops within a `luminous.launch`,
that can be added to the block with `pushBack()`. There can be more than one dispatch block within a luminous launch
op, these blocks are encapsulated in a `DispatchBlocks` container, which has the 
`addNewBlock()` method and is the only way for the user to create a new dispatch block. The pass then outlines the corresponding kernels for these blocks.

```c++
using DispatchBuilderFn = std::function<void(LaunchOp, DispatchBlocks &)>;
```
```c++
void defaultDispatchBuilderFn(LaunchOp launchOp,
                              DispatchBlocks &dispatchBlocks) {
  auto &body = launchOp.body().back();
  SmallVector<DispatchBlock> dependencies;
  for (auto &op : body) {
    if (!op.hasAttr(luminous::maxMemoryAttrName))
      continue;

    // add a new block to the dispatch blocks and fill it with ops
    auto block = dispatchBlocks.addNewBlock(dependencies);
    dependencies.clear();
    block.pushBack(&op);
    dependencies.push_back(block);
  }
}
```
`DispatchBuilderFn` is a user defined function, that is called within the luminous kernel outlining pass.
It takes the `LaunchOp` and a reference to a dispatch blocks object, that the user modifies within this function.
If no DispatchBuilderFn is provided, then the `defaultDispatchBuilderFn()` is used.
