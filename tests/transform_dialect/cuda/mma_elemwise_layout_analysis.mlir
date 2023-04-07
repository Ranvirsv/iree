#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul(%lhs : tensor<16x16xf16>, %rhs : tensor<8x16xf16>, %bias : tensor<16x8xf16>) -> tensor<16x8xf16> {
  %c0 = arith.constant 0.0 : f16
  %0 = tensor.empty() : tensor<16x8xf16>
  %1 = linalg.fill ins(%c0 : f16) outs(%0 : tensor<16x8xf16>) -> tensor<16x8xf16>
  %2 = linalg.matmul_transpose_b ins(%lhs, %rhs : tensor<16x16xf16>, tensor<8x16xf16>)
      outs(%1 : tensor<16x8xf16>) -> tensor<16x8xf16>
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types=["parallel", "parallel"]}
        ins(%2, %bias : tensor<16x8xf16>, tensor<16x8xf16>) outs(%0 : tensor<16x8xf16>) {
          ^bb0(%arg0: f16, %arg1: f16, %arg2: f16):
            %10 = arith.subf %arg0, %arg1 : f16
            %11 = math.exp %10 : f16
            linalg.yield %11 : f16
       } -> tensor<16x8xf16>
  return %3 : tensor<16x8xf16>
}

// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN:     --iree-hal-cuda-llvm-target-arch=sm_80 \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-llvmgpu-use-transform-dialect=%p/mma_elemwise_layout_analysis_codegen_spec.mlir | \
// RUN: iree-run-module --function=matmul --device=cuda \
// RUN: --input="16x16xf16=[[0.0999755859375,0.2249755859375,0.07501220703125,0.0,0.07501220703125,0.2249755859375,0.175048828125,0.07501220703125,0.175048828125,0.07501220703125,0.024993896484375,0.1500244140625,0.1500244140625,0.2249755859375,0.199951171875,0.1500244140625],[0.1500244140625,0.199951171875,0.0999755859375,0.07501220703125,0.1500244140625,0.2249755859375,0.024993896484375,0.0999755859375,0.0999755859375,0.024993896484375,0.2249755859375,0.2249755859375,0.2249755859375,0.0,0.024993896484375,0.04998779296875],[0.07501220703125,0.0,0.125,0.125,0.04998779296875,0.2249755859375,0.024993896484375,0.199951171875,0.199951171875,0.07501220703125,0.1500244140625,0.2249755859375,0.024993896484375,0.175048828125,0.07501220703125,0.125],[0.04998779296875,0.024993896484375,0.0,0.2249755859375,0.07501220703125,0.024993896484375,0.024993896484375,0.0,0.07501220703125,0.1500244140625,0.1500244140625,0.175048828125,0.2249755859375,0.1500244140625,0.07501220703125,0.0999755859375],[0.125,0.0,0.199951171875,0.04998779296875,0.199951171875,0.04998779296875,0.175048828125,0.125,0.0,0.0,0.199951171875,0.024993896484375,0.2249755859375,0.1500244140625,0.024993896484375,0.0],[0.04998779296875,0.2249755859375,0.0999755859375,0.07501220703125,0.2249755859375,0.07501220703125,0.2249755859375,0.07501220703125,0.2249755859375,0.199951171875,0.125,0.07501220703125,0.04998779296875,0.199951171875,0.125,0.1500244140625],[0.1500244140625,0.125,0.175048828125,0.04998779296875,0.125,0.1500244140625,0.1500244140625,0.125,0.0999755859375,0.0,0.199951171875,0.024993896484375,0.175048828125,0.199951171875,0.125,0.0999755859375],[0.0999755859375,0.199951171875,0.0999755859375,0.0999755859375,0.2249755859375,0.0,0.175048828125,0.0999755859375,0.125,0.07501220703125,0.07501220703125,0.175048828125,0.07501220703125,0.0,0.2249755859375,0.2249755859375],[0.07501220703125,0.024993896484375,0.199951171875,0.024993896484375,0.175048828125,0.199951171875,0.0999755859375,0.024993896484375,0.0,0.0999755859375,0.0,0.0999755859375,0.2249755859375,0.175048828125,0.0,0.0],[0.024993896484375,0.0999755859375,0.2249755859375,0.2249755859375,0.125,0.2249755859375,0.04998779296875,0.04998779296875,0.04998779296875,0.024993896484375,0.0999755859375,0.2249755859375,0.024993896484375,0.024993896484375,0.0,0.07501220703125],[0.0,0.1500244140625,0.175048828125,0.1500244140625,0.2249755859375,0.024993896484375,0.1500244140625,0.0999755859375,0.024993896484375,0.0,0.125,0.04998779296875,0.125,0.199951171875,0.024993896484375,0.199951171875],[0.024993896484375,0.04998779296875,0.199951171875,0.0,0.07501220703125,0.199951171875,0.2249755859375,0.04998779296875,0.175048828125,0.0,0.199951171875,0.199951171875,0.1500244140625,0.199951171875,0.125,0.199951171875],[0.1500244140625,0.125,0.04998779296875,0.0999755859375,0.04998779296875,0.175048828125,0.04998779296875,0.0999755859375,0.2249755859375,0.199951171875,0.125,0.1500244140625,0.0999755859375,0.07501220703125,0.07501220703125,0.0999755859375],[0.0,0.04998779296875,0.125,0.024993896484375,0.04998779296875,0.199951171875,0.04998779296875,0.0999755859375,0.199951171875,0.07501220703125,0.1500244140625,0.125,0.199951171875,0.199951171875,0.0,0.125],[0.024993896484375,0.07501220703125,0.0,0.199951171875,0.024993896484375,0.024993896484375,0.024993896484375,0.175048828125,0.04998779296875,0.04998779296875,0.04998779296875,0.07501220703125,0.07501220703125,0.1500244140625,0.175048828125,0.199951171875],[0.0,0.125,0.0,0.07501220703125,0.125,0.125,0.07501220703125,0.1500244140625,0.04998779296875,0.04998779296875,0.125,0.125,0.2249755859375,0.0999755859375,0.07501220703125,0.07501220703125]]" \
// RUN: --input="8x16xf16=[[0.175049 0.0999756 0.0249939 0.224976 0.224976 0.199951 0.150024 0.0499878 0.224976 0.0249939 0.224976 0.150024 0.125 0.150024 0.125 0.125][0.0750122 0.175049 0.199951 0.0750122 0.224976 0.150024 0.125 0.175049 0.125 0.125 0.0249939 0.0249939 0.0999756 0.224976 0.0750122 0.0249939][0.199951 0.0750122 0 0.199951 0.125 0.0249939 0.0249939 0.125 0.224976 0 0.0499878 0 0 0.0499878 0.175049 0.0999756][0 0.0499878 0.150024 0.0999756 0.175049 0.224976 0.0750122 0.175049 0.150024 0.0249939 0 0.0999756 0.0999756 0.125 0.150024 0.175049][0.175049 0.125 0.175049 0.0999756 0 0.0249939 0.125 0.175049 0 0.175049 0 0.125 0.199951 0.150024 0.175049 0.0249939][0.125 0.125 0.0999756 0.224976 0.0750122 0.150024 0.125 0.0750122 0 0.175049 0.150024 0.150024 0.125 0 0 0][0.199951 0.0750122 0.175049 0.0999756 0.0499878 0.224976 0.0750122 0.0249939 0.150024 0.0249939 0.0750122 0.224976 0.175049 0 0.0499878 0.0249939][0.0499878 0.224976 0.150024 0.0999756 0 0.199951 0.150024 0.125 0.125 0.125 0.224976 0 0.175049 0.0999756 0.125 0]]" \
// RUN: --input="16x8xf16=[[0.0,-0.03173828125,-0.1318359375,-0.044189453125,-0.0655517578125,-0.126220703125,-0.076171875,-0.041259765625],[0.0,-0.0855712890625,-0.157470703125,-0.09619140625,-0.1124267578125,-0.0718994140625,-0.04052734375,-0.0531005859375],[0.0,-0.065673828125,-0.118896484375,-0.0438232421875,-0.1031494140625,-0.1051025390625,-0.06884765625,-0.0750732421875],[0.0,-0.0911865234375,-0.11810302734375,-0.09375,-0.0711669921875,-0.06494140625,-0.083740234375,-0.0755615234375],[0.0,-0.0150146484375,-0.125,-0.064453125,-0.0462646484375,-0.065673828125,-0.064453125,-0.0325927734375],[0.0,-0.026123046875,-0.1287841796875,-0.078125,-0.1043701171875,-0.125,-0.1368408203125,-0.06005859375],[0.0,-0.036865234375,-0.1300048828125,-0.0699462890625,-0.078125,-0.11376953125,-0.088134765625,-0.03369140625],[0.0,-0.06201171875,-0.0894775390625,-0.0594482421875,-0.078857421875,-0.10693359375,-0.0982666015625,-0.0863037109375],[-0.021240234375,0.0,-0.15185546875,-0.036865234375,-0.0380859375,-0.0587158203125,-0.0343017578125,-0.045654296875],[0.0,-0.052490234375,-0.1268310546875,-0.04248046875,-0.0955810546875,-0.0399169921875,-0.029296875,-0.060546875],[0.0,-0.0162353515625,-0.1187744140625,-0.043701171875,-0.079345703125,-0.0924072265625,-0.1112060546875,-0.0599365234375],[0.0,-0.0706787109375,-0.175048828125,-0.054931640625,-0.1119384765625,-0.1356201171875,-0.076904296875,-0.06005859375],[0.0,-0.0662841796875,-0.105712890625,-0.0782470703125,-0.0838623046875,-0.078857421875,-0.061279296875,-0.0494384765625],[0.0,-0.0374755859375,-0.138671875,-0.0374755859375,-0.086181640625,-0.09619140625,-0.05615234375,-0.0318603515625],[0.0,-0.0455322265625,-0.0450439453125,-0.023681640625,-0.0343017578125,-0.07745361328125,-0.09375,-0.05126953125],[0.0,-0.041259765625,-0.11566162109375,-0.0462646484375,-0.061279296875,-0.06689453125,-0.0706787109375,-0.0325927734375]]" |\
// RUN: FileCheck %s --check-prefix=EXEC

//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 16x8xf16=[1.34863 1.34863 1.34863 1.34863 1.34863 1.34863 1.34863 1.34863][1.34766 1.34668 1.34766 1.34766 1.34668 1.34766 1.34668 1.34766][1.32715 1.32715 1.32715 1.32715 1.32715 1.32715 1.32715 1.32715][1.27246 1.27246 1.27148 1.27246 1.27148 1.27246 1.27246 1.27246][1.25586 1.25586 1.25586 1.25586 1.25586 1.25586 1.25586 1.25586][1.37598 1.37598 1.37598 1.37598 1.37598 1.37598 1.37598 1.37598][1.33984 1.33984 1.33984 1.33984 1.33984 1.33984 1.33984 1.33984][1.32715 1.32715 1.32715 1.32715 1.32715 1.32715 1.32715 1.32715][1.24023 1.24023 1.24023 1.24023 1.23926 1.24023 1.23926 1.24023][1.26855 1.26855 1.26855 1.26855 1.26855 1.26855 1.26855 1.26855][1.28516 1.28516 1.28516 1.28516 1.28516 1.28516 1.28516 1.28516][1.36523 1.36523 1.36523 1.36523 1.36523 1.36523 1.36523 1.36523][1.31348 1.31348 1.31348 1.31445 1.31445 1.31445 1.31348 1.31445][1.28027 1.28027 1.28027 1.28027 1.28027 1.28027 1.28027 1.28027][1.21387 1.21387 1.21387 1.21387 1.21387 1.21387 1.21387 1.21387][1.24902 1.24902 1.24902 1.24902 1.24902 1.24902 1.24902 1.24902]
