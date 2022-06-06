# RUN: %PYTHON %s | FileCheck %s

import ctypes
import gc, sys
import re
import itertools
from functools import reduce
from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir.dialects.linalg.opdsl.lang import *

elm_wise_template = """
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {{
  func @function(%arg0: memref<{0}>, %arg1: memref<{1}>) -> memref<{2}> {{
    %0 = memref.alloc() : memref<{0}>
    linalg.generic {{indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%arg0, %arg1 : memref<{0}>, memref<{1}>) outs(%0 : memref<{2}>) {{
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %1 = addf %arg2, %arg3 : f32
      linalg.yield %1 : f32
    }}
    return %0 : memref<{2}>
  }}
}}
"""

bcast_template = """
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
module {{
  func @function(%arg0: memref<{0}>, %arg1: memref<{1}>) -> memref<{2}> {{
    %0 = memref.alloc() : memref<{0}>
    linalg.generic {{indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%arg0, %arg1 : memref<{0}>, memref<{1}>) outs(%0 : memref<{2}>) {{
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %1 = addf %arg2, %arg3 : f32
      linalg.yield %1 : f32
    }}
    return %0 : memref<{2}>
  }}
}}
"""

# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stdout)
  sys.stderr.flush()

def run(f):
  log("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0

def transform(module, memref_size):
  import mlir.conversions
  import mlir.dialects.linalg.passes
  import mlir.transforms

  pm = PassManager.parse('builtin.func(linalg-memory-footprint-reduce{{linalg-max-memory-footprint={}}})'.format(memref_size))
  pm.run(module)
  return module

def findAllGenericOpsInOp(op):
  if isinstance(op, linalg.GenericOp):
    return [op]

  return itertools.chain.from_iterable(findAllGenericOpsInRegion(region) for region in op)

def findAllGenericOpsInBlock(block):
  return itertools.chain.from_iterable(findAllGenericOpsInOp(op) for op in block)

def findAllGenericOpsInRegion(region):
  return itertools.chain.from_iterable(findAllGenericOpsInBlock(block) for block in region)

def parseMemRefString(memrefString):
  '''
  extracts the shape and bit width of the memref string passed
  '''
  result = re.search(r"\<(.*?)\>", memrefString)
  result = result.group(1).split(',')[0].split('x')
  shape = [int(x) for x in result[:-1]]
  elementBitWidth = int(result[-1][1:])
  return shape, elementBitWidth

def calculateSize(shape, bitwidth):
  return reduce(lambda a, b: a * b, shape) * bitwidth//8

def calculateFootprintOfGenOp(genop):
  return sum(calculateSize(*parseMemRefString(str(inp.type))) for inp in genop.inputs) + \
         sum(calculateSize(*parseMemRefString(str(out.type))) for out in genop.outputs)

def transformAndCheckResults(module, maxMemSize):
  module = transform(module, maxMemSize)
  genops = findAllGenericOpsInBlock(module.body)
  success = True
  for genop in genops:
    if calculateFootprintOfGenOp(genop) > maxMemSize:
      log("FAIL")
      genop.dump()
      success = False

  if success:
    log("SUCCESS")

def testSameSizeElementWise():
  with Context() as ctx:
    args = ["64x128x1024xf32"] * 3
    maxMemSize = 100000
    module = Module.parse(elm_wise_template.format(*args))
    transformAndCheckResults(module, maxMemSize)

# CHECK-LABEL: testSameSizeElementWise
# CHECK: SUCCESS
run(testSameSizeElementWise)

def testBroadcast():
  with Context() as ctx:
    args = ["64x128x1024xf32", "128x1024xf32", "64x128x1024xf32"]
    maxMemSize = 100000
    module = Module.parse(bcast_template.format(*args))
    transformAndCheckResults(module, maxMemSize)

# CHECK-LABEL: testBroadcast
# CHECK: SUCCESS
run(testBroadcast)
