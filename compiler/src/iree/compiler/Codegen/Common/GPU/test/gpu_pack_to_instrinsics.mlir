// RUN: iree-opt %s --mlir-print-local-scope --pass-pipeline='builtin.module(func.func(iree-codegen-gpu-pack-to-intrinsics, canonicalize, cse))' --split-input-file | FileCheck %s

#config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>}>
module {
  func.func @matmul_32x32x8(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %mm = linalg.matmul {lowering_config = #config} ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %mm : tensor<64x64xf32>
  }
}

// CHECK-LABEL: func.func @matmul_32x32x8
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<64x64xf16>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<64x64xf16>
//  CHECK-SAME:   %[[C:[A-Za-z0-9]+]]: tensor<64x64xf32>
//   CHECK-DAG:   %[[A_PACK:.+]] = linalg.pack %[[A]] inner_dims_pos = [0, 1] inner_tiles = [32, 8]
//   CHECK-DAG:   %[[B_PACK:.+]] = linalg.pack %[[B]] inner_dims_pos = [1, 0] inner_tiles = [32, 8]
//   CHECK-DAG:   %[[C_PACK:.+]] = linalg.pack %[[C]] inner_dims_pos = [0, 1] inner_tiles = [32, 32]
//       CHECK:   iree_codegen.inner_tiled ins(%[[A_PACK]], %[[B_PACK]]) outs(%[[C_PACK]])
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-SAME:       affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-SAME:       affine_map<(d0, d1, d2) -> (d0, d1)>
//  CHECK-SAME:     iterator_types = {{.*}}parallel{{.*}}parallel{{.*}}reduction
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>
//  CHECK-SAME:     lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>}>
//  CHECK-SAME:     permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
module {
  func.func @matmul_16x16x16(%a: tensor<?x?x?xf16>, %b: tensor<?x?x?x?xf16>, %c: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %mm = linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
    } ins(%a, %b : tensor<?x?x?xf16>, tensor<?x?x?x?xf16>)
    outs(%c : tensor<?x?x?xf32>) attrs =  {
      lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>}>
    } {
    ^bb0(%in: f16, %in_2: f16, %out: f32):
      %4 = arith.extf %in : f16 to f32
      %5 = arith.extf %in_2 : f16 to f32
      %6 = arith.mulf %4, %5 : f32
      %7 = arith.addf %out, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<?x?x?xf32>
    return %mm : tensor<?x?x?xf32>
  }
}

// CHECK-LABEL: func.func @matmul_16x16x16
//       CHECK:   iree_codegen.inner_tiled
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d3, d4)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
//  CHECK-SAME:     lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>}>
//  CHECK-SAME:     : tensor<?x?x?x16x16xf16>, tensor<?x?x?x?x16x16xf16> into tensor<?x?x?x16x16xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @hoist_pack_unpack(%arg0 : tensor<4x8x16x4xf32>, %arg1 : tensor<8x8x16x4xf32>, %arg2 : tensor<64x127xf32>) -> tensor<64x127xf32>{
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %padding_value = arith.constant 0.000000e+00 : f32
  %1 = scf.for %arg3 = %c0 to %c512 step %c32 iter_args(%arg4 = %arg2) -> (tensor<64x127xf32>) {
      %2 = tensor.empty() : tensor<4x8x16x16xf32>
      %pack = linalg.pack %arg4 padding_value(%padding_value : f32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %2 : tensor<64x127xf32> -> tensor<4x8x16x16xf32>
      %3 = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%pack) {indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>} : tensor<4x8x16x4xf32>, tensor<8x8x16x4xf32> into tensor<4x8x16x16xf32>
      %unpack = linalg.unpack %3 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %arg4 : tensor<4x8x16x16xf32> -> tensor<64x127xf32>
      scf.yield %unpack : tensor<64x127xf32>
    }
  return %1 : tensor<64x127xf32>
}

// CHECK-LABEL: func.func @hoist_pack_unpack
// CHECK      : %[[A_PACK:.+]] = linalg.pack
// CHECK-SAME : outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16]
// CHECK      : %[[FOR_RESULT:.+]] = scf.for %arg3 = %c0 to %c512 step %c32 iter_args(%[[ARG4:.+]] = %[[A_PACK]]) -> (tensor<4x8x16x16xf32>)
// CHECK      : %[[INNER_TILED_RESULT:.+]] = iree_codegen.inner_tiled ins(%[[A_FOROP]], %[[B_FOROP]]) outs(%[[ARG4]])
// CHECK      : scf.yield %[[INNER_TILED_RESULT]] : tensor<4x8x16x16xf32>
// CHECK      : %[[EMPTY:.+]]: tensor<64x127xf32>
// CHECK      : linalg.unpack [[FOR_RESULT]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %[[EMPTY]] : tensor<4x8x16x16xf32> -> tensor<64x127xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @no_hoist_pack_unpack(%arg0 : tensor<4x8x16x4xf32>, %arg1 : tensor<8x4x16x4xf32>, %arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>{
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %1 = scf.for %arg3 = %c0 to %c512 step %c32 iter_args(%arg4 = %arg2) -> (tensor<64x64xf32>) {
      %2 = tensor.empty() : tensor<4x4x16x16xf32>
      %pack = linalg.pack %arg4 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %2 : tensor<64x64xf32> -> tensor<4x4x16x16xf32>
      %3 = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%pack) {indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>} : tensor<4x8x16x4xf32>, tensor<8x4x16x4xf32> into tensor<4x4x16x16xf32>
      %unpack = linalg.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %arg4 : tensor<4x4x16x16xf32> -> tensor<64x64xf32>
      %empty = tensor.empty() : tensor<32x16x2x4xf32>
      %pack_2 = linalg.pack %unpack inner_dims_pos = [0, 1] inner_tiles = [2, 4] into %empty : tensor<64x64xf32> -> tensor<32x16x2x4xf32>
      %empty_2 = tensor.empty() : tensor<16x32x4x2xf32>
      %transpose = linalg.transpose ins(%pack_2 : tensor<32x16x2x4xf32>) outs(%empty_2 : tensor<16x32x4x2xf32>) permutation = [1, 0, 3, 2]
      %empty_3 = tensor.empty() : tensor<64x64xf32>
      %unpack_2 = linalg.unpack %transpose outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [4, 2] into %empty_3 : tensor<16x32x4x2xf32> -> tensor<64x64xf32>
      scf.yield %unpack_2 : tensor<64x64xf32>
    }
  return %1 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @no_hoist_pack_unpack
// CHECK      : %[[FOR_RESULT:.+]] = scf.for %arg3 = %c0 to %c512 step %c32 iter_args(%[[ARG4:.+]] = %[[VAR]]) -> (tensor<64x64xf32>)
// CHECK      : %[[PACK:.+]] = linalg.pack %[[ARG4]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %[[EMPTY]] tensor<64x64xf32> -> tensor<4x4x16x16xf32>
// CHECK      : %[[INNER_TILED_RESULT:.+]] = iree_codegen.inner_tiled ins(%[[A_FOROP]], %[[B_FOROP]]) outs(%[[PACK]])
// CHECK      : %[[UNPACK:.+]] = linalg.unpack %[[TRANSPOSED]] outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [4, 2] into %[[EMPTY_0]] : tensor<16x32x4x2xf32> -> tensor<64x64xf32>
// CHECK      : scf.yield %[[UNPACK]] : tensor<4x4x16x16xf32>
