module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.mlir.global internal constant @str3("out of memory: %zu bytes for %s\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str2("__sm1\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("__mm2\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("__mm1\00") {addr_space = 0 : i32}
  func.func @k2mm_fp64_polygeist(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: f64, %arg9: f64) attributes {llvm.emit_c_interface, llvm.linkage = #llvm.linkage<external>} {
    %c8_i64 = arith.constant 8 : i64
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = llvm.mlir.addressof @str2 : !llvm.ptr
    %1 = llvm.mlir.addressof @str1 : !llvm.ptr
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = arith.index_cast %arg1 : i64 to index
    %4 = arith.muli %arg0, %arg1 : i64
    %5 = arith.muli %4, %c8_i64 : i64
    %6 = arith.index_cast %5 : i64 to index
    %7 = arith.muli %3, %c8 : index
    %8 = arith.divui %6, %7 : index
    %alloc = memref.alloc(%8, %3) : memref<?x?xf64>
    %9 = polygeist.memref2pointer %alloc : memref<?x?xf64> to !llvm.ptr
    %10 = polygeist.pointer2memref %9 : !llvm.ptr to memref<?xi8>
    %11 = polygeist.pointer2memref %2 : !llvm.ptr to memref<?xi8>
    call @__npb_alloc_check(%10, %5, %11) : (memref<?xi8>, i64, memref<?xi8>) -> ()
    %12 = arith.index_cast %arg3 : i64 to index
    %13 = arith.muli %arg0, %arg3 : i64
    %14 = arith.muli %13, %c8_i64 : i64
    %15 = arith.index_cast %14 : i64 to index
    %16 = arith.muli %12, %c8 : index
    %17 = arith.divui %15, %16 : index
    %alloc_0 = memref.alloc(%17, %12) : memref<?x?xf64>
    %18 = polygeist.memref2pointer %alloc_0 : memref<?x?xf64> to !llvm.ptr
    %19 = polygeist.pointer2memref %18 : !llvm.ptr to memref<?xi8>
    %20 = polygeist.pointer2memref %1 : !llvm.ptr to memref<?xi8>
    call @__npb_alloc_check(%19, %14, %20) : (memref<?xi8>, i64, memref<?xi8>) -> ()
    %21 = arith.index_cast %arg2 : i64 to index
    %22 = arith.muli %arg0, %arg2 : i64
    %23 = arith.muli %22, %c8_i64 : i64
    %24 = arith.index_cast %23 : i64 to index
    %25 = arith.muli %21, %c8 : index
    %26 = arith.divui %24, %25 : index
    %alloc_1 = memref.alloc(%26, %21) : memref<?x?xf64>
    %27 = polygeist.memref2pointer %alloc_1 : memref<?x?xf64> to !llvm.ptr
    %28 = polygeist.pointer2memref %27 : !llvm.ptr to memref<?xi8>
    %29 = polygeist.pointer2memref %0 : !llvm.ptr to memref<?xi8>
    call @__npb_alloc_check(%28, %23, %29) : (memref<?xi8>, i64, memref<?xi8>) -> ()
    %30 = arith.index_cast %arg0 : i64 to index
    affine.for %arg10 = 0 to %30 {
      affine.for %arg11 = 0 to %21 {
        %31 = affine.load %arg4[%arg10, %arg11] : memref<?x?xf64>
        %32 = arith.mulf %arg8, %31 : f64
        affine.store %32, %alloc_1[%arg10, %arg11] : memref<?x?xf64>
      }
    }
    affine.for %arg10 = 0 to %30 {
      affine.for %arg11 = 0 to %3 {
        affine.store %cst, %alloc[%arg10, %arg11] : memref<?x?xf64>
        affine.for %arg12 = 0 to %21 {
          %31 = affine.load %alloc_1[%arg10, %arg12] : memref<?x?xf64>
          %32 = affine.load %arg5[%arg12, %arg11] : memref<?x?xf64>
          %33 = arith.mulf %31, %32 : f64
          %34 = affine.load %alloc[%arg10, %arg11] : memref<?x?xf64>
          %35 = arith.addf %34, %33 : f64
          affine.store %35, %alloc[%arg10, %arg11] : memref<?x?xf64>
        }
      }
    }
    affine.for %arg10 = 0 to %30 {
      affine.for %arg11 = 0 to %12 {
        affine.store %cst, %alloc_0[%arg10, %arg11] : memref<?x?xf64>
        affine.for %arg12 = 0 to %3 {
          %31 = affine.load %alloc[%arg10, %arg12] : memref<?x?xf64>
          %32 = affine.load %arg6[%arg12, %arg11] : memref<?x?xf64>
          %33 = arith.mulf %31, %32 : f64
          %34 = affine.load %alloc_0[%arg10, %arg11] : memref<?x?xf64>
          %35 = arith.addf %34, %33 : f64
          affine.store %35, %alloc_0[%arg10, %arg11] : memref<?x?xf64>
        }
      }
    }
    affine.for %arg10 = 0 to %30 {
      affine.for %arg11 = 0 to %12 {
        %31 = affine.load %alloc_0[%arg10, %arg11] : memref<?x?xf64>
        %32 = affine.load %arg7[%arg10, %arg11] : memref<?x?xf64>
        %33 = arith.mulf %arg9, %32 : f64
        %34 = arith.addf %31, %33 : f64
        affine.store %34, %arg7[%arg10, %arg11] : memref<?x?xf64>
      }
    }
    memref.dealloc %alloc : memref<?x?xf64>
    memref.dealloc %alloc_0 : memref<?x?xf64>
    memref.dealloc %alloc_1 : memref<?x?xf64>
    return
  }
  func.func private @__npb_alloc_check(%arg0: memref<?xi8>, %arg1: i64, %arg2: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = llvm.mlir.addressof @str3 : !llvm.ptr
    %1 = llvm.mlir.addressof @stderr : !llvm.ptr
    %c0_i64 = arith.constant 0 : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = polygeist.memref2pointer %arg0 : memref<?xi8> to !llvm.ptr
    %4 = llvm.icmp "eq" %3, %2 : !llvm.ptr
    %5 = arith.cmpi ne, %arg1, %c0_i64 : i64
    %6 = arith.andi %4, %5 : i1
    scf.if %6 {
      %7 = llvm.load %1 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %8 = polygeist.memref2pointer %7 : memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>> to !llvm.ptr
      %9 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<33 x i8>
      %10 = polygeist.memref2pointer %arg2 : memref<?xi8> to !llvm.ptr
      %11 = llvm.call @fprintf(%8, %9, %arg1, %10) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
      func.call @abort() : () -> ()
    }
    return
  }
  func.func private @abort() attributes {llvm.linkage = #llvm.linkage<external>}
}
