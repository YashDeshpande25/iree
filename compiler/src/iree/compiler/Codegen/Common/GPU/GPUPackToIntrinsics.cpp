// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPACKTOINTRINSICSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUPackToIntrinsicsPass final
    : impl::GPUPackToIntrinsicsPassBase<GPUPackToIntrinsicsPass> {
  void runOnOperation() override;
};
} // namespace

LogicalResult packToIntrinsic(linalg::LinalgOp linalgOp,
                              RewriterBase &rewriter) {
  auto loweringConfig =
      getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
  assert(loweringConfig && "Packing unconfigured op");

  IREE::GPU::MmaInterfaceAttr kind = getMmaKind(loweringConfig);
  assert(kind && "Packing op without mma kind");

  FailureOr<linalg::ContractionDimensions> contractionDims =
      linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "failed to infer contraction dims");
  }

  if (contractionDims->m.empty() || contractionDims->n.empty() ||
      contractionDims->k.empty()) {
    return rewriter.notifyMatchFailure(
        linalgOp, "contraction like operation missing critical dimension");
  }

  auto zero = rewriter.getIndexAttr(0);
  SmallVector<OpFoldResult> packedSizes(linalgOp.getNumLoops(), zero);

  auto [m, n, k] = kind.getMNKShape();
  packedSizes[contractionDims->m.back()] = rewriter.getIndexAttr(m);
  packedSizes[contractionDims->n.back()] = rewriter.getIndexAttr(n);
  packedSizes[contractionDims->k.back()] = rewriter.getIndexAttr(k);
  FailureOr<linalg::PackResult> maybeResult =
      linalg::pack(rewriter, linalgOp, packedSizes);
  if (failed(maybeResult)) {
    return rewriter.notifyMatchFailure(linalgOp, "packing failed");
  }
  setLoweringConfig(maybeResult->packedLinalgOp, loweringConfig);
  return success();
}

struct ConvertToMultiMma final : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    auto loweringConfig =
        getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
    if (!loweringConfig) {
      return failure();
    }
    IREE::GPU::MmaInterfaceAttr kind = getMmaKind(loweringConfig);
    if (!kind) {
      return failure();
    }
    if (failed(convertContractionToInnerTiledMma(rewriter, linalgOp, kind))) {
      return failure();
    }
    return success();
  }
};

// This pattern hoists pack & unpack ops out of scf.for op.
struct PackDestinationForOp final : OpRewritePattern<scf::YieldOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::YieldOp yieldOp,
                                PatternRewriter &rewriter) const override {
    Location loc = yieldOp.getLoc();

    // Get the enclosing scf.for op.
    auto parentOp = yieldOp->getParentOp();
    auto forOp = dyn_cast<scf::ForOp>(parentOp);
    if (!forOp)
      return failure();

    linalg::UnPackOp unpackOp;
    linalg::PackOp packOp;
    int64_t tiedResultIdx = 0; // To change

    // ###############################################################################
    for (auto operand : yieldOp.getOperands()) {
      unpackOp = operand.getDefiningOp<linalg::UnPackOp>();
      // auto iterArg = forOp.getRegionIterArgs()[tiedResultIdx];
      if (!unpackOp) {
        // llvm::dbgs()<<"No unpackOp found index : "<<tiedResultIdx<<"\n";
        tiedResultIdx++;
        continue;
      }

      // Get the corresponding packOp.
      auto iterArg = forOp.getRegionIterArgs()[tiedResultIdx];
      // llvm::dbgs() << "Users of iterArg : "<<iterArg<<" = " <<
      // iterArg.getNumUses() << "\n";
      if (iterArg.getNumUses() != 2) {
        tiedResultIdx++;
        continue;
      }

      for (auto user : iterArg.getUsers()) {

        packOp = dyn_cast<linalg::PackOp>(user);
        // llvm::dbgs()<<"User of iterArg is "<<user<<"\n";
        // if(packOp)
        //   // llvm::dbgs()<<"Found a packOp\n";
        // if(!packOp)
        // llvm::dbgs()<<"Not a packOp\n";
        // ######################################################################
        //   if (packOp &&
        //     ((packOp.getInnerDimsPos() != unpackOp.getInnerDimsPos()) ||
        //      (packOp.getMixedTiles() != unpackOp.getMixedTiles()) ||
        //      (packOp.getOuterDimsPerm() != unpackOp.getOuterDimsPerm()))) {
        //   llvm::dbgs()<<"Got NO suitable packOp\n";
        //   // break;
        //   packOp = nullptr;
        // }
        // else
        //   break;
        // #######################################################################
        if (packOp &&
            ((packOp.getInnerDimsPos() == unpackOp.getInnerDimsPos()) &&
             (packOp.getMixedTiles() == unpackOp.getMixedTiles()) &&
             (packOp.getOuterDimsPerm() == unpackOp.getOuterDimsPerm()))) {
          // llvm::dbgs()<<"Got suitable packOp\n";
          break;
          // packOp = nullptr;
        } else {
          packOp = nullptr;
        }
      }
      if (packOp && unpackOp) {
        break;
      }
      // llvm::dbgs()<<"Reached end of loop, Index value :
      // "<<tiedResultIdx<<"-------\n\n";
      tiedResultIdx++;
    }

    if (!packOp || !unpackOp) {
      // llvm::dbgs()<<"PackOp or UnpackOp not found\n";
      return failure();
    }

    // llvm::dbgs()<<"starting the chain\n";
    // #######################################################################################
    // // Iterate through all the operands of yieldOp to apply pattern for the
    // unpack found first.
    // // applied recursively you will handle all unpack operands if there are
    // muliple. for(auto operand : yieldOp.getOperands())
    // {
    //   unpackOp = operand.getDefiningOp<linalg::UnPackOp>();
    //   if(unpackOp){
    //     break;
    //   }
    //   tiedResultIdx++;
    // }
    // if(!unpackOp)
    //   return failure();

    // auto iterArg = forOp.getRegionIterArgs()[tiedResultIdx];
    // llvm::dbgs()<<"iterArg is "<<iterArg<<"\n";
    // llvm::dbgs()<<"iterArg has 1 use : "<<iterArg.hasOneUse()<<"\n";
    // // Get the corresponding packOp.
    //   for (auto user : iterArg.getUsers()) {
    //     packOp = dyn_cast<linalg::PackOp>(user);
    //     // Both the packOp & unpackOp should have the same packing
    //     attributes. if (packOp &&
    //        ((packOp.getInnerDimsPos() != unpackOp.getInnerDimsPos()) ||
    //         (packOp.getMixedTiles() != unpackOp.getMixedTiles()) ||
    //         (packOp.getOuterDimsPerm() != unpackOp.getOuterDimsPerm()))) {
    //         packOp = nullptr;
    //       }
    //   }

    // // No unpack/packOp op to hoist out.
    // if (!unpackOp || !packOp)
    //   return failure();
    // ########################################################################
    // Location loc = yieldOp.getLoc();
    // linalg::UnPackOp unpackOp;
    // if (yieldOp.getNumOperands())
    //   unpackOp = yieldOp.getOperand(0).getDefiningOp<linalg::UnPackOp>();

    // // No unpack op to hoist out.
    // if (!unpackOp)
    //   return failure();

    // // Get the enclosing scf.for op.
    // auto parentOp = yieldOp->getParentOp();
    // auto forOp = dyn_cast<scf::ForOp>(parentOp);
    // if (!forOp)
    //   return failure();

    // Get the packOp corresponding to the unpackOp.
    // linalg::PackOp packOp;
    // for (auto user : forOp.getRegionIterArgs()[0].getUsers()) {
    //   packOp = dyn_cast<linalg::PackOp>(user);

    //   // Both the packOp & unpackOp should have the same packing attributes.
    //   if (packOp &&
    //       ((packOp.getInnerDimsPos() != unpackOp.getInnerDimsPos()) ||
    //        (packOp.getMixedTiles() != unpackOp.getMixedTiles()) ||
    //        (packOp.getOuterDimsPerm() != unpackOp.getOuterDimsPerm()))) {
    //     packOp = nullptr;
    //   }
    // }
    // if (!packOp)
    //   return failure();
    // #################################################################################

    // Create the pack -> new scf.for -> unpack chain.
    rewriter.setInsertionPoint(forOp);
    Value input = linalg::PackOp::createDestinationTensor(
        rewriter, loc, forOp.getInitArgs()[tiedResultIdx],
        packOp.getMixedTiles(), packOp.getInnerDimsPos(),
        packOp.getOuterDimsPerm());

    auto packedDest = rewriter.create<linalg::PackOp>(
        loc, forOp.getInitArgs()[tiedResultIdx], input,
        packOp.getInnerDimsPos(), packOp.getMixedTiles(),
        packOp.getPaddingValue(), packOp.getOuterDimsPerm());

    // llvm::dbgs()<<"New packOp created\n";

    // auto packOpValues = ValueRange{packedDest};
    auto packOpValues = llvm::to_vector_of<Value>(forOp.getInitArgs());
    // for (auto some_Value : packOpValues)
    // {
    //   // llvm::dbgs()<<"value from the new PackOp : "<<some_Value<<"\n";
    // }

    // llvm::dbgs()<<"Result of the new packOp :
    // "<<packedDest.getResult()<<"\n";

    packOpValues[tiedResultIdx] = packedDest.getResult();

    // for (auto initArg : packOpValues)
    // {
    //   // llvm::dbgs()<<"initIterArg : "<<initArg<<"\n";
    // }

    // llvm::dbgs()<<"ValueRange : "<<ValueRange{packedDest}<<"\n";
    // scf::ForOp newForOp = rewriter.create<scf::ForOp>(
    //     loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
    //     ValueRange{packedDest});
    scf::ForOp newForOp = rewriter.create<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        packOpValues);

    // llvm::dbgs()<<"forOp created\n";
    // Destination tensor for the new unpackOp, based on the shape of the
    // original tensor that got packed, to help unpack into unaligned shapes and
    // drop padding added by the packOp.
    Value empty = rewriter.create<tensor::EmptyOp>(
        loc, packOp.getSourceType().getShape(),
        packOp.getSourceType().getElementType());

    auto unpackedOutput = rewriter.create<linalg::UnPackOp>(
        loc, newForOp.getResults()[tiedResultIdx], empty,
        unpackOp.getInnerDimsPos(), unpackOp.getMixedTiles(),
        unpackOp.getOuterDimsPerm());

    // llvm::dbgs()<<"packOp created\n";
    // Users of the result of unpackOp must use the input to the unpackOp.
    unpackOp->getResult(0).replaceAllUsesWith(unpackOp.getOperand(0));

    // Users of the result of packOp must use the init of the forOp.
    for (auto user : forOp.getRegionIterArgs()[tiedResultIdx].getUsers()) {
      user->getResult(0).replaceAllUsesWith(
          newForOp.getRegionIterArgs()[tiedResultIdx]);
    }

    // llvm::dbgs()<<"Replaced uses\n";

    // Merge the old scf.for block with the new scf.for block.
    SmallVector<Value> ivs = {newForOp.getInductionVar()};
    SmallVector<Value> argReplacements(ivs);
    // llvm::dbgs()<<"created the vectors\n";
    argReplacements.append(newForOp.getRegionIterArgs().begin(),
                           newForOp.getRegionIterArgs().end());

    // llvm::dbgs()<<"Append completed\n";
    rewriter.mergeBlocks(forOp.getBody(), newForOp.getBody(), argReplacements);
    // llvm::dbgs()<<"Merged Blocks\n";

    // Replaces the uses of the old scf.for with the new scf.for.
    for (int idx = 0; idx < forOp->getNumResults(); ++idx) {
      if (idx == tiedResultIdx) {
        forOp->getResult(idx).replaceAllUsesWith(unpackedOutput->getResult(0));
      } else {
        // llvm::dbgs()<<"Entered last else case\n";
        forOp->getResult(idx).replaceAllUsesWith(newForOp->getResult(idx));
      }
    }
    // llvm::dbgs()<<"Return Success\n";
    return success();
  }
};

void GPUPackToIntrinsicsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  // Step 1. Pack candidate linalg ops to specified shapes.
  IRRewriter rewriter(funcOp);
  SmallVector<linalg::LinalgOp> packingCandidates;
  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    auto loweringConfig =
        getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
    if (!loweringConfig) {
      return;
    }
    if (!getMmaKind(loweringConfig)) {
      return;
    }
    packingCandidates.push_back(linalgOp);
  });

  for (auto candidate : packingCandidates) {
    rewriter.setInsertionPoint(candidate);
    if (failed(packToIntrinsic(candidate, rewriter))) {
      funcOp.emitError() << "failed to pack operation marked with intrinsic\n";
      return signalPassFailure();
    }
  }

  // Step 2. Convert configured linalg ops to inner_tiled ops with multi-MMA
  // intrinsic kinds.
  {
    RewritePatternSet patterns(context);
    patterns.add<ConvertToMultiMma>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError() << "failed to convert linalg to multi-MMA inner_tiled";
      return signalPassFailure();
    }
  }

  // Step 3. Run layout propagation patterns to pull in adjacent un-configured
  // ops.
  RewritePatternSet patterns(context);
  linalg::ControlPropagationFn control = [](OpOperand *opOperand) -> bool {
    Operation *producer = opOperand->get().getDefiningOp();
    Operation *consumer = opOperand->getOwner();
    return !getLoweringConfig(producer) && !getLoweringConfig(consumer);
  };

  linalg::populateDataLayoutPropagationPatterns(patterns, control);
  patterns.add<PackDestinationForOp>(context);
  linalg::UnPackOp::getCanonicalizationPatterns(patterns, context);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
