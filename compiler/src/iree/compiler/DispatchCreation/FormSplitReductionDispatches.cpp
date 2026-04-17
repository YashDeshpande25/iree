// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/DispatchCreation/Passes.h"

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-form-split-reduction-dispatches"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FORMSPLITREDUCTIONDISPATCHESPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

struct FormSplitReductionDispatchesPass final
    : impl::FormSplitReductionDispatchesPassBase<
          FormSplitReductionDispatchesPass> {
  using Base::Base;
  void runOnOperation() override;

private:
  std::optional<SmallVector<OpFoldResult>>
  getUserSpecifiedTileSize(PartialReductionOpInterface op) const;
};
} // namespace

namespace {

  /// Fuses two sibling `linalg.reduce` ops that share reduction dimensions
  /// and iteration space into a single multi-output `linalg.reduce`.
  ///
  /// This enables the case where split-reduction turned a multi-output
  /// linalg.generic into N separate linalg.reduce ops (one per output).
  /// FormDispatchRegions treats each as its own root => N dispatches.
  /// After this fusion, there is one root => one dispatch.
  struct FuseSiblingLinalgReducePattern
      : public OpRewritePattern<linalg::ReduceOp> {
    using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;
  
    LogicalResult matchAndRewrite(linalg::ReduceOp firstOp,
                                  PatternRewriter &rewriter) const override {
      // Skip if already inside a dispatch region.
      if (firstOp->getParentOfType<IREE::Flow::DispatchRegionOp>() ||
          firstOp->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
        return failure();
      }
  
      // Find a sibling linalg.reduce in the same block that can be fused.
      Block *parentBlock = firstOp->getBlock();
      linalg::ReduceOp secondOp = nullptr;
      for (Operation &op : *parentBlock) {
        auto candidate = dyn_cast<linalg::ReduceOp>(&op);
        if (!candidate || candidate == firstOp) continue;
        if (!areFusable(firstOp, candidate)) continue;
        if (hasDependencyBetween(firstOp, candidate)) continue;
        secondOp = candidate;
        break;
      }
      if (!secondOp) return failure();
  
      // Build the fused multi-output linalg.reduce.
      SmallVector<Value> fusedInputs(firstOp.getInputs());
      fusedInputs.append(secondOp.getInputs().begin(),
                         secondOp.getInputs().end());
      SmallVector<Value> fusedInits(firstOp.getInits());
      fusedInits.append(secondOp.getInits().begin(),
                        secondOp.getInits().end());
  
      // Pick the op that dominates to be the insertion point.
      Operation *insertionOp =
          firstOp->isBeforeInBlock(secondOp) ? secondOp.getOperation()
                                             : firstOp.getOperation();
      rewriter.setInsertionPoint(insertionOp);
  
      auto fusedOp = linalg::ReduceOp::create(
          rewriter, firstOp.getLoc(), fusedInputs, fusedInits,
          firstOp.getDimensions(),
          [&](OpBuilder &b, Location loc, ValueRange args) {
            // args layout: [in0_a, in0_b, ..., init0_a, init0_b, ...]
            unsigned n1 = firstOp.getInputs().size();
            unsigned n2 = secondOp.getInputs().size();
            unsigned numIns = n1 + n2;
  
            // Clone first op's combiner, remapping its block args.
            IRMapping map1;
            for (unsigned i = 0; i < n1; ++i) {
              map1.map(firstOp.getCombiner().getArgument(i), args[i]);
              map1.map(firstOp.getCombiner().getArgument(n1 + i),
                       args[numIns + i]);
            }
            SmallVector<Value> yields1;
            for (Operation &op :
                 firstOp.getCombiner().front().without_terminator()) {
              b.clone(op, map1);
            }
            auto yield1 = cast<linalg::YieldOp>(
                firstOp.getCombiner().front().getTerminator());
            for (Value v : yield1.getValues())
              yields1.push_back(map1.lookup(v));
  
            // Clone second op's combiner, remapping its block args.
            IRMapping map2;
            for (unsigned i = 0; i < n2; ++i) {
              map2.map(secondOp.getCombiner().getArgument(i), args[n1 + i]);
              map2.map(secondOp.getCombiner().getArgument(n2 + i),
                       args[numIns + n1 + i]);
            }
            SmallVector<Value> yields2;
            for (Operation &op :
                 secondOp.getCombiner().front().without_terminator()) {
              b.clone(op, map2);
            }
            auto yield2 = cast<linalg::YieldOp>(
                secondOp.getCombiner().front().getTerminator());
            for (Value v : yield2.getValues())
              yields2.push_back(map2.lookup(v));
  
            SmallVector<Value> allYields(yields1);
            allYields.append(yields2.begin(), yields2.end());
            linalg::YieldOp::create(b, loc, allYields);
          });
  
      // Replace uses: first op's results are fusedOp.getResults()[0..n1),
      // second op's results are fusedOp.getResults()[n1..n1+n2).
      unsigned n1 = firstOp.getResults().size();
      rewriter.replaceOp(firstOp, fusedOp.getResults().take_front(n1));
      rewriter.replaceOp(secondOp, fusedOp.getResults().drop_front(n1));
      return success();
    }
  
  private:
    static bool areFusable(linalg::ReduceOp a, linalg::ReduceOp b) {
      if (a.getDimensions() != b.getDimensions()) return false;
  
      // All inputs in the fused op must share a shape (SameVariadicOperandSize
      // + shape check in verifier).
      auto aInType = cast<ShapedType>(a.getInputs()[0].getType());
      auto bInType = cast<ShapedType>(b.getInputs()[0].getType());
      if (aInType.getShape() != bInType.getShape()) return false;
  
      // Same for inits.
      auto aInitType = cast<ShapedType>(a.getInits()[0].getType());
      auto bInitType = cast<ShapedType>(b.getInits()[0].getType());
      if (aInitType.getShape() != bInitType.getShape()) return false;
  
      return true;
    }
  
    static bool hasDependencyBetween(linalg::ReduceOp a, linalg::ReduceOp b) {
      // Reject if b uses any result of a, or vice versa. A proper impl should
      // also check transitive deps via backward slice; this is the simple check.
      for (Value result : a->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (user == b.getOperation()) return true;
        }
      }
      for (Value result : b->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (user == a.getOperation()) return true;
        }
      }
      return false;
    }
  };
  
  } // namespace

static SmallVector<unsigned> getReductionDims(TilingInterface op) {
  SmallVector<unsigned> dims;
  for (auto [i, loopType] : llvm::enumerate(op.getLoopIteratorTypes())) {
    if (loopType == utils::IteratorType::reduction) {
      dims.push_back(i);
    }
  }
  return dims;
}

static FailureOr<IREE::Flow::DispatchRegionOp>
tileOpAndWrapInDispatch(RewriterBase &rewriter, TilingInterface op,
                        ArrayRef<OpFoldResult> splitSize, bool fusePad) {
  IRRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  scf::SCFTilingOptions options;
  options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  options.setReductionTilingStrategy(
      ReductionTilingStrategy::PartialReductionOuterParallel);

  // Set tile sizes.
  SmallVector<OpFoldResult> tileSizes;
  auto zeroAttr = rewriter.getIndexAttr(0);
  int splitSizeIndex = 0;
  for (utils::IteratorType iteratorType : op.getLoopIteratorTypes()) {
    if (iteratorType == utils::IteratorType::parallel) {
      tileSizes.push_back(zeroAttr);
    } else {
      tileSizes.push_back(splitSize[splitSizeIndex++]);
    }
  }
  options.setTileSizes(tileSizes);
  SmallVector<unsigned> reductionDims = getReductionDims(op);
  auto mapping = llvm::map_to_vector(
      llvm::seq<int64_t>(0, reductionDims.size()),
      [&](int64_t index) -> Attribute {
        return IREE::LinalgExt::SplitReductionMappingAttr::get(
            rewriter.getContext(), reductionDims.size() - 1 - index);
      });
  options.setReductionDims(reductionDims);
  options.setMapping(mapping);

  // Tile the operation and fuse with producers.
  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  // Only fuse along the dest operand.
  scf::SCFTileAndFuseOptions::ControlFnTy fusionControlFn =
      [](tensor::ExtractSliceOp extractOp, OpResult result, bool isDestArg)
      -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
    if (isDestArg) {
      return scf::SCFTileAndFuseOptions::ControlFnResult{false};
    }
    Operation *extractSource = extractOp.getSource().getDefiningOp();
    if (extractSource && IREE::LinalgExt::isBitExtendOp(extractSource)) {
      return scf::SCFTileAndFuseOptions::ControlFnResult{false};
    }
    return std::nullopt;
  };
  tileAndFuseOptions.setFusionControlFn(fusionControlFn);
  tileAndFuseOptions.setTilingOptions(std::move(options));

  MLIRContext *context = rewriter.getContext();
  RewritePatternSet cleanupPatterns(context);
  populateFoldExtractSliceOfBroadcastPattern(cleanupPatterns);
  if (fusePad) {
    // When fusing pads we do not want to generate zeroSliceGuards.
    cleanupPatterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
        context,
        [](tensor::ExtractSliceOp) { return /*zeroSliceGuard=*/false; });
  }
  tileAndFuseOptions.cleanupPatterns =
      FrozenRewritePatternSet(std::move(cleanupPatterns));

  FailureOr<scf::SCFTileAndFuseResult> result =
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, op,
                                                tileAndFuseOptions);
  if (failed(result)) {
    return op.emitOpError("failed to tile using scf.forall");
  }
  for (auto [origValue, replacement] : result->replacements) {
    rewriter.replaceAllUsesWith(origValue, replacement);
  }

  // Didn't tile.
  if (result->loops.size() == 0) {
    return success();
  }
  assert(result->loops.size() == 1 &&
         "expected to get a single loop after tiling");

  // Wrap loop in `flow.dispatch.region`.
  LoopLikeOpInterface loop = result->loops[0];
  FailureOr<IREE::Flow::DispatchRegionOp> maybeRegionOp =
      IREE::Flow::wrapOpInDispatchRegion(rewriter, loop);
  if (failed(maybeRegionOp)) {
    return loop.emitOpError("failed to wrap in dispatch region");
  }
  return maybeRegionOp.value();
}

std::optional<SmallVector<OpFoldResult>>
FormSplitReductionDispatchesPass::getUserSpecifiedTileSize(
    PartialReductionOpInterface op) const {
  {
    // First preference given to attribute set on the op.
    std::optional<SmallVector<int64_t>> attributeTileSize =
        IREE::LinalgExt::getSplitReductionSizes(op);
    if (attributeTileSize) {
      MLIRContext *context = op->getContext();
      return getAsIndexOpFoldResult(context, attributeTileSize.value());
    }
  }

  // Use the pass option as the next lever. This is mostly used for testing.
  if (!splitSize.empty()) {
    unsigned numReduction = llvm::count_if(
        op.getLoopIteratorTypes(), [](utils::IteratorType iteratorType) {
          return iteratorType == utils::IteratorType::reduction;
        });
    if (numReduction == 0) {
      return std::nullopt;
    }
    SmallVector<int64_t> tileSizes(numReduction, 0);
    for (auto [index, tileSize] : llvm::enumerate(llvm::reverse(splitSize))) {
      tileSizes[numReduction - 1 - index] = tileSize;
    }
    MLIRContext *context = op->getContext();
    return getAsIndexOpFoldResult(context, tileSizes);
  }

  // Default.
  return std::nullopt;
}

void FormSplitReductionDispatchesPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  SmallVector<std::pair<PartialReductionOpInterface, SmallVector<OpFoldResult>>>
      reductionOps;
  funcOp.walk([&](PartialReductionOpInterface tilingOp) {
    std::optional<SmallVector<OpFoldResult>> tileSizes =
        getUserSpecifiedTileSize(tilingOp);
    if (!tileSizes) {
      return;
    }
    reductionOps.emplace_back(tilingOp, std::move(tileSizes.value()));
  });

  if (reductionOps.empty()) {
    // Nothing to do.
    return;
  }

  for (auto [op, tileSizes] : reductionOps) {
    FailureOr<IREE::Flow::DispatchRegionOp> formedDispatch =
        tileOpAndWrapInDispatch(rewriter, op, tileSizes, enableFusePad);
    if (failed(formedDispatch)) {
      op->emitOpError("failed to form split reduction dispatch");
      return signalPassFailure();
    }
  }

  // Run some canonicalization patterns within dispatches.
  RewritePatternSet patterns(context);
  linalg::populateSwapExtractSliceWithFillPatterns(patterns);
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  patterns.insert<FuseSiblingLinalgReducePattern>(context);  // <-- NEW
  GreedyRewriteConfig config;
  config.setMaxIterations(GreedyRewriteConfig::kNoLimit).enableFolding(true);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
    funcOp.emitOpError("failed to apply tiling canonicalization patterns");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
