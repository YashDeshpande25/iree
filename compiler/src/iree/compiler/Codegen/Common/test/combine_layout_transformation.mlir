// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-combine-layout-transformation,canonicalize,cse))" -split-input-file %s | FileCheck %s

func.func @fold_collapse_shape_op(%source : tensor<2x4x16xf32>, %result : memref<8x16xf32>) {
  %collapse = tensor.collapse_shape %source [[0, 1], [2]] : tensor<2x4x16xf32> into tensor<8x16xf32>
  iree_codegen.store_to_buffer %collapse, %result : tensor<8x16xf32> into memref<8x16xf32>
  return
}
// CHECK-LABEL: @fold_collapse_shape_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<8x16xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
//       CHECK:     %[[LINEARIZE:.+]] = affine.linearize_index
//  CHECK-SAME:       [%[[IDX0]], %[[IDX1]]] by (2, 4)
//       CHECK:     iree_linalg_ext.yield %[[LINEARIZE]], %[[IDX2]], %[[TRUE]]
//       CHECK:   } : tensor<2x4x16xf32> into tensor<8x16xf32> -> tensor<8x16xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<8x16xf32> into memref<8x16xf32>

// -----

func.func @fold_expand_shape_op(%source : tensor<8x16xf32>, %result : memref<2x4x16xf32>) {
  %expand = tensor.expand_shape %source [[0, 1], [2]] output_shape [2, 4, 16] : tensor<8x16xf32> into tensor<2x4x16xf32>
  iree_codegen.store_to_buffer %expand, %result : tensor<2x4x16xf32> into memref<2x4x16xf32>
  return
}
// CHECK-LABEL: @fold_expand_shape_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<2x4x16xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       CHECK:     %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[IDX0]] into (2, 4)
//       CHECK:     iree_linalg_ext.yield %[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1, %[[IDX1]], %[[TRUE]]
//       CHECK:   } : tensor<8x16xf32> into tensor<2x4x16xf32> -> tensor<2x4x16xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<2x4x16xf32> into memref<2x4x16xf32>

// -----

func.func @fold_transpose_op(%source : tensor<2x4x16xf32>, %result : memref<4x16x2xf32>) {
  %init = tensor.empty() : tensor<4x16x2xf32>
  %transposed = linalg.transpose ins(%source : tensor<2x4x16xf32>) outs(%init : tensor<4x16x2xf32>) permutation = [1, 2, 0]
  iree_codegen.store_to_buffer %transposed, %result : tensor<4x16x2xf32> into memref<4x16x2xf32>
  return
}
// CHECK-LABEL: @fold_transpose_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<4x16x2xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
//       CHECK:     iree_linalg_ext.yield %[[IDX1]], %[[IDX2]], %[[IDX0]], %[[TRUE]]
//       CHECK:   } : tensor<2x4x16xf32> into tensor<4x16x2xf32> -> tensor<4x16x2xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<4x16x2xf32> into memref<4x16x2xf32>

// -----

func.func @fold_extract_slice_op(%source : tensor<64xf32>, %result : memref<63xf32>) {
  %slice = tensor.extract_slice %source[0] [63] [1] : tensor<64xf32> to tensor<63xf32>
  iree_codegen.store_to_buffer %slice, %result : tensor<63xf32> into memref<63xf32>
  return
}
// CHECK-LABEL: @fold_extract_slice_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C63:.+]] = arith.constant 63 : index
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<63xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index):
//       CHECK:     %[[MASK:.+]] = arith.cmpi ult, %[[IDX0]], %[[C63]]
//       CHECK:     iree_linalg_ext.yield %[[IDX0]], %[[MASK]]
//       CHECK:   } : tensor<64xf32> into tensor<63xf32> -> tensor<63xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<63xf32> into memref<63xf32>

// -----

func.func @no_fold_offset_extract_slice_op(%source : tensor<64xf32>, %result : memref<4xf32>) {
  %slice = tensor.extract_slice %source[42] [4] [1] : tensor<64xf32> to tensor<4xf32>
  iree_codegen.store_to_buffer %slice, %result : tensor<4xf32> into memref<4xf32>
  return
}
// CHECK-LABEL: @no_fold_offset_extract_slice_op
//       CHECK:   tensor.extract_slice
//   CHECK-NOT:   iree_linalg_ext.map_scatter

// -----

func.func @no_fold_strided_extract_slice_op(%source : tensor<64xf32>, %result : memref<16xf32>) {
  %slice = tensor.extract_slice %source[0] [16] [4] : tensor<64xf32> to tensor<16xf32>
  iree_codegen.store_to_buffer %slice, %result : tensor<16xf32> into memref<16xf32>
  return
}
// CHECK-LABEL: @no_fold_strided_extract_slice_op
//       CHECK:   tensor.extract_slice
//   CHECK-NOT:   iree_linalg_ext.map_scatter

// -----

func.func @fold_pad_op(%source : tensor<250xf32>, %result : memref<256xf32>) {
  %cst = arith.constant 0.0 : f32
  %padded = tensor.pad %source low[2] high[4] {
  ^bb0(%arg0: index):
    tensor.yield %cst : f32
  } : tensor<250xf32> to tensor<256xf32>
  iree_codegen.store_to_buffer %padded, %result : tensor<256xf32> into memref<256xf32>
  return
}
//       CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (256, d0 + 64)>
// CHECK-LABEL: @fold_pad_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[PAD_VAL:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C252:.+]] = arith.constant 252 : index
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<256xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index):
//       CHECK:     iree_linalg_ext.yield %[[IDX0]], %[[TRUE]]
//       CHECK:   } : tensor<250xf32> into tensor<256xf32> -> tensor<256xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<256xf32> into memref<256xf32>

//       CHECK:   scf.forall (%[[WG_IV:.+]]) = (0) to (256) step (64) {
//       CHECK:     %[[WG_TILE_UB:.+]] = affine.min #[[$MAP]](%[[WG_IV]])
//       CHECK:     scf.for %[[IDX:.+]] = %[[WG_IV]] to %[[WG_TILE_UB]] step %[[C1]] {
//   CHECK-DAG:       %[[IS_LOW_PAD:.+]] = arith.cmpi ult, %[[IDX]], %[[C2]]
//   CHECK-DAG:       %[[IS_HIGH_PAD:.+]] = arith.cmpi uge, %[[IDX]], %[[C252]]
//   CHECK-DAG:       %[[IS_PAD:.+]] = arith.ori %[[IS_LOW_PAD]], %[[IS_HIGH_PAD]] : i1
//       CHECK:       scf.if %[[IS_PAD]] {
//  CHECK-NEXT:         memref.store %[[PAD_VAL]], %[[RESULT]][%[[IDX]]] : memref<256xf32>
//  CHECK-NEXT:       }
//  CHECK:          }
//  CHECK:        } {mapping = [#iree_codegen.workgroup_mapping<x>]}

// -----

func.func @fold_unpack_op(%source : tensor<?x?x128x128xf32>, %result : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = memref.dim %result, %c0 : memref<?x?xf32>
  %d1 = memref.dim %result, %c1 : memref<?x?xf32>
  %dest = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %unpack = linalg.unpack %source
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 128]
      into %dest : tensor<?x?x128x128xf32> -> tensor<?x?xf32>
  iree_codegen.store_to_buffer %unpack, %result : tensor<?x?xf32> into memref<?x?xf32>
  return
}
//       CHECK: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 * 128)>
// CHECK-LABEL: @fold_unpack_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[RES_D0:.+]] = memref.dim %[[RESULT]], %[[C0]] : memref<?x?xf32>
//   CHECK-DAG:   %[[RES_D1:.+]] = memref.dim %[[RESULT]], %[[C1]] : memref<?x?xf32>
//   CHECK-DAG:   %[[SRC_D0:.+]] = tensor.dim %[[SOURCE]], %[[C0]] : tensor<?x?x128x128xf32>
//   CHECK-DAG:   %[[SRC_D1:.+]] = tensor.dim %[[SOURCE]], %[[C1]] : tensor<?x?x128x128xf32>
//   CHECK-DAG:   %[[COLLAPSE_SIZE0:.+]] = affine.apply #[[$MAP]]()[%[[SRC_D0]]]
//   CHECK-DAG:   %[[COLLAPSE_SIZE1:.+]] = affine.apply #[[$MAP]]()[%[[SRC_D1]]]
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty(%[[RES_D0]], %[[RES_D1]]) : tensor<?x?xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index, %[[IDX3:.+]]: index):
//       CHECK:     %[[LINEARIZE:.+]] = affine.linearize_index
//  CHECK-SAME:       [%[[IDX0]], %[[IDX2]], %[[IDX1]], %[[IDX3]]]
//  CHECK-SAME:       by (%[[SRC_D0]], 128, %[[SRC_D1]], 128)
//       CHECK:     %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[LINEARIZE]]
//  CHECK-SAME:       into (%[[COLLAPSE_SIZE0]], %[[COLLAPSE_SIZE1]])
//       CHECK:     %[[BOUND0:.+]] = arith.cmpi ult, %[[DELINEARIZE]]#0, %[[RES_D0]]
//       CHECK:     %[[BOUND1:.+]] = arith.cmpi ult, %[[DELINEARIZE]]#1, %[[RES_D1]]
//       CHECK:     %[[MASK:.+]] = arith.andi %[[BOUND0]], %[[BOUND1]] : i1
//       CHECK:     iree_linalg_ext.yield %[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1, %[[MASK]]
//       CHECK:   } : tensor<?x?x128x128xf32> into tensor<?x?xf32> -> tensor<?x?xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<?x?xf32> into memref<?x?xf32>

// -----

func.func @fold_pack_op(%source : tensor<250x250xf32>, %result : memref<2x2x128x128xf32>) {
  %cst = arith.constant 0.0 : f32
  %dest = tensor.empty() : tensor<2x2x128x128xf32>
  %unpack = linalg.pack %source padding_value(%cst : f32)
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 128]
      into %dest : tensor<250x250xf32> -> tensor<2x2x128x128xf32>
  iree_codegen.store_to_buffer %unpack, %result : tensor<2x2x128x128xf32> into memref<2x2x128x128xf32>
  return
}
//       CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (256, d0 + 1)>
//       CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (256, d0 + 64)>
// CHECK-LABEL: @fold_pack_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
//   CHECK-DAG:   %[[PAD_VAL:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C250:.+]] = arith.constant 250 : index
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<2x2x128x128xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       CHECK:     %[[DELINEARIZE0:.+]]:2 = affine.delinearize_index %[[IDX0]]
//  CHECK-SAME:       into (2, 128)
//       CHECK:     %[[DELINEARIZE1:.+]]:2 = affine.delinearize_index %[[IDX1]]
//  CHECK-SAME:       into (2, 128)
//       CHECK:     iree_linalg_ext.yield %[[DELINEARIZE0]]#0, %[[DELINEARIZE1]]#0, %[[DELINEARIZE0]]#1, %[[DELINEARIZE1]]#1, %[[TRUE]]
//       CHECK:   } : tensor<250x250xf32> into tensor<2x2x128x128xf32> -> tensor<2x2x128x128xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<2x2x128x128xf32> into memref<2x2x128x128xf32>

//       CHECK:   scf.forall (%[[WG_IV0:.+]], %[[WG_IV1:.+]]) = (0, 0) to (256, 256) step (1, 64) {
//   CHECK-DAG:     %[[WG_TILE_UB0:.+]] = affine.min #[[$MAP]](%[[WG_IV0]])
//   CHECK-DAG:     %[[WG_TILE_UB1:.+]] = affine.min #[[$MAP1]](%[[WG_IV1]])
//       CHECK:     scf.for %[[IDX0:.+]] = %[[WG_IV0]] to %[[WG_TILE_UB0]] step %[[C1]] {
//       CHECK:       scf.for %[[IDX1:.+]] = %[[WG_IV1]] to %[[WG_TILE_UB1]] step %[[C1]] {
//   CHECK-DAG:         %[[EXPANDED_IDX0:.+]]:2 = affine.delinearize_index %[[IDX0]] into (2, 128)
//   CHECK-DAG:         %[[EXPANDED_IDX1:.+]]:2 = affine.delinearize_index %[[IDX1]] into (2, 128)
//   CHECK-DAG:         %[[IDX0_IS_LOW_PAD:.+]] = arith.cmpi ult, %[[IDX0]], %[[C0]]
//   CHECK-DAG:         %[[IDX0_IS_HIGH_PAD:.+]] = arith.cmpi uge, %[[IDX0]], %[[C250]]
//   CHECK-DAG:         %[[IDX0_IS_PAD:.+]] = arith.ori %[[IDX0_IS_LOW_PAD]], %[[IDX0_IS_HIGH_PAD]] : i1
//   CHECK-DAG:         %[[IDX1_IS_LOW_PAD:.+]] = arith.cmpi ult, %[[IDX1]], %[[C0]]
//   CHECK-DAG:         %[[IDX1_IS_HIGH_PAD:.+]] = arith.cmpi uge, %[[IDX1]], %[[C250]]
//   CHECK-DAG:         %[[IDX1_IS_PAD:.+]] = arith.ori %[[IDX1_IS_LOW_PAD]], %[[IDX1_IS_HIGH_PAD]] : i1
//   CHECK-DAG:         %[[IS_PAD:.+]] = arith.ori %[[IDX0_IS_PAD]], %[[IDX1_IS_PAD]] : i1
//       CHECK:         scf.if %[[IS_PAD]] {
//  CHECK-NEXT:           memref.store %[[PAD_VAL]], %[[RESULT]]
//  CHECK-SAME:             [%[[EXPANDED_IDX0]]#0, %[[EXPANDED_IDX1]]#0, %[[EXPANDED_IDX0]]#1, %[[EXPANDED_IDX1]]#1]
//  CHECK-SAME:             : memref<2x2x128x128xf32>
//  CHECK-NEXT:         }
//  CHECK:            }
//  CHECK:          }
//  CHECK:        } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

func.func @propagate_relayout_ops(%source : tensor<?x?x128x128xf32>,
                                  %result : memref<?xf16>,
                                  %size0: index, %size1: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dest = tensor.empty(%size0, %size1) : tensor<?x?xf32>
  %unpack = linalg.unpack %source
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 128]
      into %dest : tensor<?x?x128x128xf32> -> tensor<?x?xf32>
  %collapse = tensor.collapse_shape %unpack [[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
  %flat_size = arith.muli %size0, %size1 : index
  %init = tensor.empty(%flat_size) : tensor<?xf16>
  %compute_op = linalg.generic
      {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
       iterator_types = ["parallel"]}
       ins(%collapse : tensor<?xf32>) outs(%init : tensor<?xf16>) {
  ^bb0(%in: f32, %out: f16):
    %trunc = arith.truncf %in : f32 to f16
    linalg.yield %trunc : f16
  } -> tensor<?xf16>
  iree_codegen.store_to_buffer %compute_op, %result : tensor<?xf16> into memref<?xf16>
  return
}
// CHECK-LABEL: @propagate_relayout_ops
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//       CHECK:   %[[INIT:.+]] = tensor.empty{{.*}} : tensor<?x?x128x128xf16>
//       CHECK:   %[[COMPUTE_OP:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[SOURCE]] : tensor<?x?x128x128xf32>)
//  CHECK-SAME:     outs(%[[INIT]] : tensor<?x?x128x128xf16>)
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter %[[COMPUTE_OP]]
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]]


// -----

// #####################################################################
func.func @check_workgroup_mapping(%2 : tensor<196608x35xbf16>, %9 : tensor<8x16x1x16xbf16>) -> tensor<196608x35xbf16>{
  // %cst = arith.constant 0.000000e+00 : bf16
  // %c0 = arith.constant 0 : index
  // %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196608x35xbf16>>
  
  // %5 = tensor.empty() : tensor<196608x35xbf16>
  %6 = scf.forall (%arg0, %arg1) = (0, 0) to (196608, 35) step (128, 16) shared_outs(%arg2 = %2) -> (tensor<196608x35xbf16>) {
    // %7 = affine.min affine_map<(d0) -> (-d0 + 35, 16)>(%arg1)
    
    // %8 = tensor.empty() : tensor<8x16x1x16xbf16>
    // %9 = linalg.fill ins(%cst : bf16) outs(%8 : tensor<8x16x1x16xbf16>) -> tensor<8x16x1x16xbf16>

    // %13 = tensor.empty() : tensor<8x16x1x16xbf16>
    // %extracted_slice = tensor.extract_slice %arg2[%arg0, %arg1] [128, %7] [1, 1] : tensor<196608x35xbf16> to tensor<128x?xbf16>
    // %transposed = linalg.transpose ins(%9 : tensor<8x1x16x16xbf16>) outs(%13 : tensor<8x16x1x16xbf16>) permutation = [0, 2, 1, 3] 
    %collapsed = tensor.collapse_shape %9 [[0, 1], [2, 3]] : tensor<8x16x1x16xbf16> into tensor<128x16xbf16>
    // %extracted_slice_1 = tensor.extract_slice %collapsed[0, 0] [128, %7] [1, 1] : tensor<128x16xbf16> to tensor<128x?xbf16>
    // %14 = linalg.copy ins(%extracted_slice_1 : tensor<128x?xbf16>) outs(%extracted_slice : tensor<128x?xbf16>) -> tensor<128x?xbf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %collapsed into %arg2[%arg0, %arg1] [128, 16] [1, 1] : tensor<128x16xbf16> into tensor<196608x35xbf16>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  // {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  // scf.forall (%arg0, %arg1) = (%0, %1) to (%2, %3) step(%4, %5) {
  //         "use"(%arg0, %arg1) : (index, index) -> ()
  //         %collapsed = tensor.collapse_shape %transposed [[0, 1], [2, 3]] : tensor<8x16x1x16xbf16> into tensor<128x16xbf16>
  //         scf.forall.in_parallel {}
  //       } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  // iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [196608, 35], strides = [1, 1] : tensor<196608x35xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196608x35xbf16>>
  return %6 : tensor<196608x35xbf16>
}

// CHECK-LABEL: @check_workgroup_mapping
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//       CHECK:        } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

// Test to look at an scf.forall op & not insert map_scatter as it is not a workgroup mapping
func.func @check_no_workgroup_mapping(%3: tensor<64x64xf32>, %4: tensor<64x64xf32>, %5: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {}>
    } {
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %6 = scf.for %arg0 = %c0 to %c64 step %c8 iter_args(%arg1 = %5) -> (tensor<64x64xf32>) {
    %extracted_slice = tensor.extract_slice %3[0, %arg0] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
    %extracted_slice_0 = tensor.extract_slice %4[0, %arg0] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
    %extracted_slice_1 = tensor.extract_slice %arg1[0, %arg0] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
    
    %1 = scf.forall (%arg5, %arg6) = (0, 0) to (64, 8) step (1, 4) shared_outs(%arg7 = %extracted_slice_1) -> (tensor<64x8xf32>) {
      %extracted_slice_2 = tensor.extract_slice %extracted_slice[%arg5, %arg6] [1, 4] [1, 1] : tensor<64x8xf32> to tensor<1x4xf32>
      %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg5, %arg6] [1, 4] [1, 1] : tensor<64x8xf32> to tensor<1x4xf32>
      %extracted_slice_4 = tensor.extract_slice %arg7[%arg5, %arg6] [1, 4] [1, 1] : tensor<64x8xf32> to tensor<1x4xf32>
      %2 = linalg.add ins(%extracted_slice_2, %extracted_slice_3 : tensor<1x4xf32>, tensor<1x4xf32>) outs(%extracted_slice_4 : tensor<1x4xf32>) -> tensor<1x4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %2 into %arg7[%arg5, %arg6] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<64x8xf32>
      }
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
    %7 = linalg.add
      ins(%1, %1 : tensor<64x8xf32>, tensor<64x8xf32>)
      outs(%1 : tensor<64x8xf32>) -> tensor<64x8xf32>
    %insert = tensor.insert_slice %7 into %arg1[0, %arg0] [64, 8] [1, 1] : tensor<64x8xf32> into tensor<64x64xf32>
    scf.yield %insert : tensor<64x64xf32>
  }
  // %8 = linalg.add
  //   ins(%6, %6 : tensor<64x64xf32>, tensor<64x64xf32>)
  //   outs(%6 : tensor<64x64xf32>) -> tensor<64x64xf32>
  // return %8 : tensor<64x64xf32>
  return %6 : tensor<64x64xf32>
}

// CHECK-LABEL: @check_no_workgroup_mapping
//   CHECK-NOT:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//   CHECK-NOT:        } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}







// func.func @check_no_work_groupmapping(){
//   %21 = tensor.empty() : tensor<16x16xbf16>
//   %22 = scf.forall (%arg5, %arg6) = (0, 0) to (16, 16) step (1, 2) shared_outs(%arg7 = %21) -> (tensor<16x16xbf16>) {
//         %24 = affine.min affine_map<(d0, d1, d2) -> (16, d1 - d2, d0)>(%arg5, %7, %19)
//         %25 = affine.min affine_map<(d0, d1) -> (1, d0 - d1)>(%20, %24)
//         %26 = affine.apply affine_map<(d0) -> (-d0 + 1)>(%25)
//         %27 = affine.min affine_map<(d0, d1) -> (-d1 + 35, 16, d0)>(%arg6, %15)
//         %28 = affine.min affine_map<(d0, d1) -> (2, d0 - d1)>(%16, %27)
//         %29 = affine.apply affine_map<(d0) -> (-d0 + 2)>(%28)
//         %30 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%24, %19, %arg1]
//         %31 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%27, %15]
//         %extracted_slice_3 = tensor.extract_slice %4[%30, %31] [%25, %28] [1, 1] : tensor<35x35xbf16> to tensor<?x?xbf16>
//         %padded = tensor.pad %extracted_slice_3 low[0, 0] high[%26, %29] {
//         ^bb0(%arg8: index, %arg9: index):
//           tensor.yield %cst : bf16
//         } : tensor<?x?xbf16> to tensor<1x2xbf16>
//         %extracted_slice_4 = tensor.extract_slice %arg7[%arg5, %arg6] [1, 2] [1, 1] : tensor<16x16xbf16> to tensor<1x2xbf16>
//         %32 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%padded : tensor<1x2xbf16>) outs(%extracted_slice_4 : tensor<1x2xbf16>) -> tensor<1x2xbf16>
//         scf.forall.in_parallel {
//           tensor.parallel_insert_slice %32 into %arg7[%arg5, %arg6] [1, 2] [1, 1] : tensor<1x2xbf16> into tensor<16x16xbf16>
//         }
//       } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
// }


// #####################################################################



// func.func @conv_2d_bfloat16_forward_128x48x32x35_nhwc_35x1x1x35_fhwc_nhwf_1x1s_0x0p_1x1d_1g$async_dispatch_0_matmul_like_196608x35x35_bf16xbf16xf32() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
//   %c3 = arith.constant 3 : index
//   %c1 = arith.constant 1 : index
//   %cst = arith.constant 0.000000e+00 : bf16
//   %cst_0 = arith.constant 0.000000e+00 : f32
//   %c0 = arith.constant 0 : index
//   %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196608x35xbf16>>
//   %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<35x35xbf16>>
//   %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196608x35xbf16>>
//   %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [196608, 35], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196608x35xbf16>> -> tensor<196608x35xbf16>
//   %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [35, 35], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<35x35xbf16>> -> tensor<35x35xbf16>
//   %5 = tensor.empty() : tensor<196608x35xbf16>
//   %6 = scf.forall (%arg0, %arg1) = (0, 0) to (196608, 35) step (128, 16) shared_outs(%arg2 = %5) -> (tensor<196608x35xbf16>) {
//     %7 = affine.min affine_map<(d0) -> (-d0 + 35, 16)>(%arg1)
//     %8 = tensor.empty() : tensor<8x1x16x16xf32>
//     %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<8x1x16x16xf32>) -> tensor<8x1x16x16xf32>
//     %10 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %9) -> (tensor<8x1x16x16xf32>) {
//       %15 = affine.min affine_map<(d0) -> (35, d0 * 16)>(%arg3)
//       %16 = affine.min affine_map<(d0) -> (-d0 + 35, 16)>(%15)
//       %17 = affine.apply affine_map<(d0) -> (-d0 + 16)>(%16)
//       %extracted_slice_2 = tensor.extract_slice %3[%arg0, %15] [128, %16] [1, 1] : tensor<196608x35xbf16> to tensor<128x?xbf16>
//       %padded = tensor.pad %extracted_slice_2 low[0, 0] high[0, %17] {
//       ^bb0(%arg5: index, %arg6: index):
//         tensor.yield %cst : bf16
//       } : tensor<128x?xbf16> to tensor<128x16xbf16>
//       %18 = tensor.empty() : tensor<128x16xbf16>
//       %19 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%padded : tensor<128x16xbf16>) outs(%18 : tensor<128x16xbf16>) -> tensor<128x16xbf16>
//       %20 = tensor.empty() : tensor<8x1x16x16xbf16>
//       %expanded = tensor.expand_shape %19 [[0, 1], [2, 3]] output_shape [8, 16, 1, 16] : tensor<128x16xbf16> into tensor<8x16x1x16xbf16>
//       %transposed_3 = linalg.transpose ins(%expanded : tensor<8x16x1x16xbf16>) outs(%20 : tensor<8x1x16x16xbf16>) permutation = [0, 2, 1, 3] 
//       %21 = affine.min affine_map<(d0) -> (-d0 + 35, 0, 16)>(%arg1)
//       %22 = affine.min affine_map<(d0, d1) -> (16, d0 - d1)>(%7, %21)
//       %23 = affine.apply affine_map<(d0) -> (-d0 + 16)>(%22)
//       %24 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%21, %arg1]
//       %extracted_slice_4 = tensor.extract_slice %4[%24, %15] [%22, %16] [1, 1] : tensor<35x35xbf16> to tensor<?x?xbf16>
//       %padded_5 = tensor.pad %extracted_slice_4 low[0, 0] high[%23, %17] {
//       ^bb0(%arg5: index, %arg6: index):
//         tensor.yield %cst : bf16
//       } : tensor<?x?xbf16> to tensor<16x16xbf16>
//       %25 = tensor.empty() : tensor<16x16xbf16>
//       %26 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%padded_5 : tensor<16x16xbf16>) outs(%25 : tensor<16x16xbf16>) -> tensor<16x16xbf16>
//       %27 = tensor.empty() : tensor<1x1x16x16xbf16>
//       %expanded_6 = tensor.expand_shape %26 [[0, 1], [2, 3]] output_shape [1, 16, 1, 16] : tensor<16x16xbf16> into tensor<1x16x1x16xbf16>
//       %transposed_7 = linalg.transpose ins(%expanded_6 : tensor<1x16x1x16xbf16>) outs(%27 : tensor<1x1x16x16xbf16>) permutation = [0, 2, 1, 3] 
//       %28 = iree_codegen.inner_tiled ins(%transposed_3, %transposed_7) outs(%arg4) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding = [128, 16, 16], promote_operands = [0, 1], reduction = [0, 0, 1], subgroup = [4, 1, 0], workgroup = [128, 16, 0]}>, permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]} : tensor<8x1x16x16xbf16>, tensor<1x1x16x16xbf16> into tensor<8x1x16x16xf32>
//       scf.yield %28 : tensor<8x1x16x16xf32>
//     }
//     %extracted_slice = tensor.extract_slice %arg2[%arg0, %arg1] [128, %7] [1, 1] : tensor<196608x35xbf16> to tensor<128x?xbf16>
//     %11 = tensor.empty() : tensor<8x1x16x16xbf16>
//     %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<8x1x16x16xf32>) outs(%11 : tensor<8x1x16x16xbf16>) {
//     ^bb0(%in: f32, %out: bf16):
//       %15 = arith.truncf %in : f32 to bf16
//       linalg.yield %15 : bf16
//     } -> tensor<8x1x16x16xbf16>
//     %13 = tensor.empty() : tensor<8x16x1x16xbf16>
//     %transposed = linalg.transpose ins(%12 : tensor<8x1x16x16xbf16>) outs(%13 : tensor<8x16x1x16xbf16>) permutation = [0, 2, 1, 3] 
//     %collapsed = tensor.collapse_shape %transposed [[0, 1], [2, 3]] : tensor<8x16x1x16xbf16> into tensor<128x16xbf16>
//     %extracted_slice_1 = tensor.extract_slice %collapsed[0, 0] [128, %7] [1, 1] : tensor<128x16xbf16> to tensor<128x?xbf16>
//     %14 = linalg.copy ins(%extracted_slice_1 : tensor<128x?xbf16>) outs(%extracted_slice : tensor<128x?xbf16>) -> tensor<128x?xbf16>
//     scf.forall.in_parallel {
//       tensor.parallel_insert_slice %14 into %arg2[%arg0, %arg1] [128, %7] [1, 1] : tensor<128x?xbf16> into tensor<196608x35xbf16>
//     }
//   } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
//   iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [196608, 35], strides = [1, 1] : tensor<196608x35xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196608x35xbf16>>
//   return
// }