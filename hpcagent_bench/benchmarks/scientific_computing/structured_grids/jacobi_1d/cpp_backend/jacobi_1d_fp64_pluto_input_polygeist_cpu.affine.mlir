#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d_fp64_polygeist(%arg0: i64, %arg1: i64, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.emit_c_interface, llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.333300e-01 : f64
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.index_cast %arg1 : i64 to index
    affine.for %arg4 = 1 to %1 {
      affine.for %arg5 = 1 to #map()[%0] {
        %2 = affine.load %arg2[%arg5 - 1] : memref<?xf64>
        %3 = affine.load %arg2[%arg5] : memref<?xf64>
        %4 = arith.addf %2, %3 : f64
        %5 = affine.load %arg2[%arg5 + 1] : memref<?xf64>
        %6 = arith.addf %4, %5 : f64
        %7 = arith.mulf %6, %cst : f64
        affine.store %7, %arg3[%arg5] : memref<?xf64>
      }
      affine.for %arg5 = 1 to #map()[%0] {
        %2 = affine.load %arg3[%arg5 - 1] : memref<?xf64>
        %3 = affine.load %arg3[%arg5] : memref<?xf64>
        %4 = arith.addf %2, %3 : f64
        %5 = affine.load %arg3[%arg5 + 1] : memref<?xf64>
        %6 = arith.addf %4, %5 : f64
        %7 = arith.mulf %6, %cst : f64
        affine.store %7, %arg2[%arg5] : memref<?xf64>
      }
    }
    return
  }
}
