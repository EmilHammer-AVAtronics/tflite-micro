/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_BUILTIN_OPS_H_
#define TENSORFLOW_LITE_BUILTIN_OPS_H_

// DO NOT EDIT MANUALLY: This file is automatically generated by
// `schema/builtin_ops_header/generator.cc`.

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// The enum for builtin operators.
// Note: CUSTOM, DELEGATE, and PLACEHOLDER_FOR_GREATER_OP_CODES are 3 special
// ops which are not real built-in ops.
typedef enum {
  kTfLiteBuiltinAdd = 0,
  kTfLiteBuiltinAveragePool2d = 1,
  kTfLiteBuiltinConcatenation = 2,
  kTfLiteBuiltinConv2d = 3,
  kTfLiteBuiltinDepthwiseConv2d = 4,
  kTfLiteBuiltinDepthToSpace = 5,
  kTfLiteBuiltinDequantize = 6,
  kTfLiteBuiltinEmbeddingLookup = 7,
  kTfLiteBuiltinFloor = 8,
  kTfLiteBuiltinFullyConnected = 9,
  kTfLiteBuiltinHashtableLookup = 10,
  kTfLiteBuiltinL2Normalization = 11,
  kTfLiteBuiltinL2Pool2d = 12,
  kTfLiteBuiltinLocalResponseNormalization = 13,
  kTfLiteBuiltinLogistic = 14,
  kTfLiteBuiltinLshProjection = 15,
  kTfLiteBuiltinLstm = 16,
  kTfLiteBuiltinMaxPool2d = 17,
  kTfLiteBuiltinMul = 18,
  kTfLiteBuiltinRelu = 19,
  kTfLiteBuiltinReluN1To1 = 20,
  kTfLiteBuiltinRelu6 = 21,
  kTfLiteBuiltinReshape = 22,
  kTfLiteBuiltinResizeBilinear = 23,
  kTfLiteBuiltinRnn = 24,
  kTfLiteBuiltinSoftmax = 25,
  kTfLiteBuiltinSpaceToDepth = 26,
  kTfLiteBuiltinSvdf = 27,
  kTfLiteBuiltinTanh = 28,
  kTfLiteBuiltinConcatEmbeddings = 29,
  kTfLiteBuiltinSkipGram = 30,
  kTfLiteBuiltinCall = 31,
  kTfLiteBuiltinCustom = 32,
  kTfLiteBuiltinEmbeddingLookupSparse = 33,
  kTfLiteBuiltinPad = 34,
  kTfLiteBuiltinUnidirectionalSequenceRnn = 35,
  kTfLiteBuiltinGather = 36,
  kTfLiteBuiltinBatchToSpaceNd = 37,
  kTfLiteBuiltinSpaceToBatchNd = 38,
  kTfLiteBuiltinTranspose = 39,
  kTfLiteBuiltinMean = 40,
  kTfLiteBuiltinSub = 41,
  kTfLiteBuiltinDiv = 42,
  kTfLiteBuiltinSqueeze = 43,
  kTfLiteBuiltinUnidirectionalSequenceLstm = 44,
  kTfLiteBuiltinStridedSlice = 45,
  kTfLiteBuiltinBidirectionalSequenceRnn = 46,
  kTfLiteBuiltinExp = 47,
  kTfLiteBuiltinTopkV2 = 48,
  kTfLiteBuiltinSplit = 49,
  kTfLiteBuiltinLogSoftmax = 50,
  kTfLiteBuiltinDelegate = 51,
  kTfLiteBuiltinBidirectionalSequenceLstm = 52,
  kTfLiteBuiltinCast = 53,
  kTfLiteBuiltinPrelu = 54,
  kTfLiteBuiltinMaximum = 55,
  kTfLiteBuiltinArgMax = 56,
  kTfLiteBuiltinMinimum = 57,
  kTfLiteBuiltinLess = 58,
  kTfLiteBuiltinNeg = 59,
  kTfLiteBuiltinPadv2 = 60,
  kTfLiteBuiltinGreater = 61,
  kTfLiteBuiltinGreaterEqual = 62,
  kTfLiteBuiltinLessEqual = 63,
  kTfLiteBuiltinSelect = 64,
  kTfLiteBuiltinSlice = 65,
  kTfLiteBuiltinSin = 66,
  kTfLiteBuiltinTransposeConv = 67,
  kTfLiteBuiltinSparseToDense = 68,
  kTfLiteBuiltinTile = 69,
  kTfLiteBuiltinExpandDims = 70,
  kTfLiteBuiltinEqual = 71,
  kTfLiteBuiltinNotEqual = 72,
  kTfLiteBuiltinLog = 73,
  kTfLiteBuiltinSum = 74,
  kTfLiteBuiltinSqrt = 75,
  kTfLiteBuiltinRsqrt = 76,
  kTfLiteBuiltinShape = 77,
  kTfLiteBuiltinPow = 78,
  kTfLiteBuiltinArgMin = 79,
  kTfLiteBuiltinFakeQuant = 80,
  kTfLiteBuiltinReduceProd = 81,
  kTfLiteBuiltinReduceMax = 82,
  kTfLiteBuiltinPack = 83,
  kTfLiteBuiltinLogicalOr = 84,
  kTfLiteBuiltinOneHot = 85,
  kTfLiteBuiltinLogicalAnd = 86,
  kTfLiteBuiltinLogicalNot = 87,
  kTfLiteBuiltinUnpack = 88,
  kTfLiteBuiltinReduceMin = 89,
  kTfLiteBuiltinFloorDiv = 90,
  kTfLiteBuiltinReduceAny = 91,
  kTfLiteBuiltinSquare = 92,
  kTfLiteBuiltinZerosLike = 93,
  kTfLiteBuiltinFill = 94,
  kTfLiteBuiltinFloorMod = 95,
  kTfLiteBuiltinRange = 96,
  kTfLiteBuiltinResizeNearestNeighbor = 97,
  kTfLiteBuiltinLeakyRelu = 98,
  kTfLiteBuiltinSquaredDifference = 99,
  kTfLiteBuiltinMirrorPad = 100,
  kTfLiteBuiltinAbs = 101,
  kTfLiteBuiltinSplitV = 102,
  kTfLiteBuiltinUnique = 103,
  kTfLiteBuiltinCeil = 104,
  kTfLiteBuiltinReverseV2 = 105,
  kTfLiteBuiltinAddN = 106,
  kTfLiteBuiltinGatherNd = 107,
  kTfLiteBuiltinCos = 108,
  kTfLiteBuiltinWhere = 109,
  kTfLiteBuiltinRank = 110,
  kTfLiteBuiltinElu = 111,
  kTfLiteBuiltinReverseSequence = 112,
  kTfLiteBuiltinMatrixDiag = 113,
  kTfLiteBuiltinQuantize = 114,
  kTfLiteBuiltinMatrixSetDiag = 115,
  kTfLiteBuiltinRound = 116,
  kTfLiteBuiltinHardSwish = 117,
  kTfLiteBuiltinIf = 118,
  kTfLiteBuiltinWhile = 119,
  kTfLiteBuiltinNonMaxSuppressionV4 = 120,
  kTfLiteBuiltinNonMaxSuppressionV5 = 121,
  kTfLiteBuiltinScatterNd = 122,
  kTfLiteBuiltinSelectV2 = 123,
  kTfLiteBuiltinDensify = 124,
  kTfLiteBuiltinSegmentSum = 125,
  kTfLiteBuiltinBatchMatmul = 126,
  kTfLiteBuiltinPlaceholderForGreaterOpCodes = 127,
  kTfLiteBuiltinCumsum = 128,
  kTfLiteBuiltinCallOnce = 129,
  kTfLiteBuiltinBroadcastTo = 130,
  kTfLiteBuiltinRfft2d = 131,
  kTfLiteBuiltinConv3d = 132,
  kTfLiteBuiltinImag = 133,
  kTfLiteBuiltinReal = 134,
  kTfLiteBuiltinComplexAbs = 135,
  kTfLiteBuiltinHashtable = 136,
  kTfLiteBuiltinHashtableFind = 137,
  kTfLiteBuiltinHashtableImport = 138,
  kTfLiteBuiltinHashtableSize = 139,
  kTfLiteBuiltinReduceAll = 140,
  kTfLiteBuiltinConv3dTranspose = 141,
  kTfLiteBuiltinVarHandle = 142,
  kTfLiteBuiltinReadVariable = 143,
  kTfLiteBuiltinAssignVariable = 144,
  kTfLiteBuiltinBroadcastArgs = 145,
  kTfLiteBuiltinRandomStandardNormal = 146,
  kTfLiteBuiltinBucketize = 147,
  kTfLiteBuiltinRandomUniform = 148,
  kTfLiteBuiltinMultinomial = 149,
  kTfLiteBuiltinGelu = 150,
  kTfLiteBuiltinDynamicUpdateSlice = 151,
  kTfLiteBuiltinRelu0To1 = 152,
  kTfLiteBuiltinUnsortedSegmentProd = 153,
  kTfLiteBuiltinUnsortedSegmentMax = 154,
  kTfLiteBuiltinUnsortedSegmentSum = 155,
  kTfLiteBuiltinAtan2 = 156,
  kTfLiteBuiltinUnsortedSegmentMin = 157,
  kTfLiteBuiltinSign = 158,
  kTfLiteBuiltinBitcast = 159,
  kTfLiteBuiltinBitwiseXor = 160,
  kTfLiteBuiltinRightShift = 161,
  kTfLiteBuiltinStablehloLogistic = 162,
  kTfLiteBuiltinStablehloAdd = 163,
  kTfLiteBuiltinStablehloDivide = 164,
  kTfLiteBuiltinStablehloMultiply = 165,
  kTfLiteBuiltinStablehloMaximum = 166,
  kTfLiteBuiltinStablehloReshape = 167,
  kTfLiteBuiltinStablehloClamp = 168,
  kTfLiteBuiltinStablehloConcatenate = 169,
  kTfLiteBuiltinStablehloBroadcastInDim = 170,
  kTfLiteBuiltinStablehloConvolution = 171,
  kTfLiteBuiltinStablehloSlice = 172,
  kTfLiteBuiltinStablehloCustomCall = 173,
  kTfLiteBuiltinStablehloReduce = 174,
  kTfLiteBuiltinStablehloAbs = 175,
  kTfLiteBuiltinStablehloAnd = 176,
  kTfLiteBuiltinStablehloCosine = 177,
  kTfLiteBuiltinStablehloExponential = 178,
  kTfLiteBuiltinStablehloFloor = 179,
  kTfLiteBuiltinStablehloLog = 180,
  kTfLiteBuiltinStablehloMinimum = 181,
  kTfLiteBuiltinStablehloNegate = 182,
  kTfLiteBuiltinStablehloOr = 183,
  kTfLiteBuiltinStablehloPower = 184,
  kTfLiteBuiltinStablehloRemainder = 185,
  kTfLiteBuiltinStablehloRsqrt = 186,
  kTfLiteBuiltinStablehloSelect = 187,
  kTfLiteBuiltinStablehloSubtract = 188,
  kTfLiteBuiltinStablehloTanh = 189,
  kTfLiteBuiltinStablehloScatter = 190,
  kTfLiteBuiltinStablehloCompare = 191,
  kTfLiteBuiltinStablehloConvert = 192,
  kTfLiteBuiltinStablehloDynamicSlice = 193,
  kTfLiteBuiltinStablehloDynamicUpdateSlice = 194,
  kTfLiteBuiltinStablehloPad = 195,
  kTfLiteBuiltinStablehloIota = 196,
  kTfLiteBuiltinStablehloDotGeneral = 197,
  kTfLiteBuiltinStablehloReduceWindow = 198,
  kTfLiteBuiltinStablehloSort = 199,
  kTfLiteBuiltinStablehloWhile = 200,
  kTfLiteBuiltinStablehloGather = 201,
  kTfLiteBuiltinStablehloTranspose = 202,
  kTfLiteBuiltinDilate = 203,
  kTfLiteBuiltinStablehloRngBitGenerator = 204,
  kTfLiteBuiltinReduceWindow = 205,
  kTfLiteBuiltinStablehloComposite = 206,
} TfLiteBuiltinOperator;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_LITE_BUILTIN_OPS_H_
