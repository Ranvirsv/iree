## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines JAX models."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions

JAX_MODELS_ROOT_DIR = "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.10_1684283564"

# Derived from https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel.
BERT_LARGE_1X384_FP32_JAX = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_1X384_FP32_JAX,
    name="BertLargeJAXBatch1",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{JAX_MODELS_ROOT_DIR}/BERT_LARGE/batch_1/stablehlo.mlirbc",
    entry_function="main",
    input_types=["1x384xi32", "1x384xi32"])

BERT_LARGE_16X384_FP32_JAX = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_16X384_FP32_JAX,
    name="BertLargeJAXBatch16",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-16"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{JAX_MODELS_ROOT_DIR}/BERT_LARGE/batch_16/stablehlo.mlirbc",
    entry_function="main",
    input_types=["16x384xi32", "16x384xi32"])

BERT_LARGE_24X384_FP32_JAX = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_24X384_FP32_JAX,
    name="BertLargeJAXBatch24",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-24"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{JAX_MODELS_ROOT_DIR}/BERT_LARGE/batch_24/stablehlo.mlirbc",
    entry_function="main",
    input_types=["24x384xi32", "24x384xi32"])

BERT_LARGE_32X384_FP32_JAX = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_32X384_FP32_JAX,
    name="BertLargeJAXBatch32",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{JAX_MODELS_ROOT_DIR}/BERT_LARGE/batch_32/stablehlo.mlirbc",
    entry_function="main",
    input_types=["32x384xi32", "32x384xi32"])

BERT_LARGE_48X384_FP32_JAX = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_48X384_FP32_JAX,
    name="BertLargeJAXBatch48",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-48"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{JAX_MODELS_ROOT_DIR}/BERT_LARGE/batch_48/stablehlo.mlirbc",
    entry_function="main",
    input_types=["48x384xi32", "48x384xi32"])

BERT_LARGE_64X384_FP32_JAX = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_64X384_FP32_JAX,
    name="BertLargeJAXBatch64",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-64"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{JAX_MODELS_ROOT_DIR}/BERT_LARGE/batch_64/stablehlo.mlirbc",
    entry_function="main",
    input_types=["64x384xi32", "64x384xi32"])

BERT_LARGE_512X384_FP32_JAX = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_512X384_FP32_JAX,
    name="BertLargeJAXBatch512",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-512"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{JAX_MODELS_ROOT_DIR}/BERT_LARGE/batch_512/stablehlo.mlirbc",
    entry_function="main",
    input_types=["512x384xi32", "512x384xi32"])

BERT_LARGE_1024X384_FP32_JAX = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_1024X384_FP32_JAX,
    name="BertLargeJAXBatch1024",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-1024"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{JAX_MODELS_ROOT_DIR}/BERT_LARGE/batch_1024/stablehlo.mlirbc",
    entry_function="main",
    input_types=["1024x384xi32", "1024x384xi32"])

BERT_LARGE_1280X384_FP32_JAX = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_1280X384_FP32_JAX,
    name="BertLargeJAXBatch1280",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-1280"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{JAX_MODELS_ROOT_DIR}/BERT_LARGE/batch_1280/stablehlo.mlirbc",
    entry_function="main",
    input_types=["1280x384xi32", "1280x384xi32"])
