// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IREECODEGENINTERFACES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IREECODEGENINTERFACES_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenEnums.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"

// clang-format off
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h.inc"
// clang-format on

#endif // IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IREECODEGENINTERFACES_H_
