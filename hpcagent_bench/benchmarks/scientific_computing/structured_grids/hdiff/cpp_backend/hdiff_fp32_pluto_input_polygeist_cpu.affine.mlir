#map = affine_map<()[s0] -> (s0 + 2)>
#map1 = affine_map<()[s0] -> (s0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.mlir.global internal constant @str7("out of memory: %zu bytes for %s\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str6("fly_field\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("res2\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("flx_field\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("res1\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("lap_field\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("__cb2\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("__cb1\00") {addr_space = 0 : i32}
  func.func @hdiff_fp32_polygeist(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: memref<?x?x?xf32>, %arg4: memref<?x?x?xf32>, %arg5: memref<?x?x?xf32>) attributes {llvm.emit_c_interface, llvm.linkage = #llvm.linkage<external>} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.mlir.addressof @str0 : !llvm.ptr
    %1 = llvm.mlir.addressof @str1 : !llvm.ptr
    %2 = llvm.mlir.addressof @str2 : !llvm.ptr
    %3 = llvm.mlir.addressof @str3 : !llvm.ptr
    %4 = llvm.mlir.addressof @str4 : !llvm.ptr
    %5 = llvm.mlir.addressof @str5 : !llvm.ptr
    %6 = llvm.mlir.addressof @str6 : !llvm.ptr
    %cst = arith.constant 4.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c4_i64 = arith.constant 4 : i64
    %7 = arith.index_cast %arg0 : i64 to index
    %8 = arith.index_cast %arg1 : i64 to index
    %9 = arith.index_cast %arg2 : i64 to index
    %10 = arith.muli %8, %9 : index
    %11 = arith.addi %arg0, %c1_i64 : i64
    %12 = arith.muli %11, %arg1 : i64
    %13 = arith.muli %12, %arg2 : i64
    %14 = arith.muli %13, %c4_i64 : i64
    %15 = arith.index_cast %14 : i64 to index
    %16 = arith.muli %10, %c4 : index
    %17 = arith.divui %15, %16 : index
    %alloc = memref.alloc(%17, %8, %9) : memref<?x?x?xf32>
    %18 = polygeist.memref2pointer %alloc : memref<?x?x?xf32> to !llvm.ptr
    %19 = polygeist.pointer2memref %18 : !llvm.ptr to memref<?xi8>
    %20 = polygeist.pointer2memref %0 : !llvm.ptr to memref<?xi8>
    call @__npb_alloc_check(%19, %14, %20) : (memref<?xi8>, i64, memref<?xi8>) -> ()
    %21 = arith.addi %arg1, %c1_i64 : i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = arith.muli %22, %9 : index
    %24 = arith.muli %arg0, %21 : i64
    %25 = arith.muli %24, %arg2 : i64
    %26 = arith.muli %25, %c4_i64 : i64
    %27 = arith.index_cast %26 : i64 to index
    %28 = arith.muli %23, %c4 : index
    %29 = arith.divui %27, %28 : index
    %alloc_1 = memref.alloc(%29, %22, %9) : memref<?x?x?xf32>
    %30 = polygeist.memref2pointer %alloc_1 : memref<?x?x?xf32> to !llvm.ptr
    %31 = polygeist.pointer2memref %30 : !llvm.ptr to memref<?xi8>
    %32 = polygeist.pointer2memref %1 : !llvm.ptr to memref<?xi8>
    call @__npb_alloc_check(%31, %26, %32) : (memref<?xi8>, i64, memref<?xi8>) -> ()
    %33 = arith.addi %arg1, %c2_i64 : i64
    %34 = arith.index_cast %33 : i64 to index
    %35 = arith.muli %34, %9 : index
    %36 = arith.addi %arg0, %c2_i64 : i64
    %37 = arith.muli %36, %33 : i64
    %38 = arith.muli %37, %arg2 : i64
    %39 = arith.muli %38, %c4_i64 : i64
    %40 = arith.index_cast %39 : i64 to index
    %41 = arith.muli %35, %c4 : index
    %42 = arith.divui %40, %41 : index
    %alloc_2 = memref.alloc(%42, %34, %9) : memref<?x?x?xf32>
    %43 = polygeist.memref2pointer %alloc_2 : memref<?x?x?xf32> to !llvm.ptr
    %44 = polygeist.pointer2memref %43 : !llvm.ptr to memref<?xi8>
    %45 = polygeist.pointer2memref %2 : !llvm.ptr to memref<?xi8>
    call @__npb_alloc_check(%44, %39, %45) : (memref<?xi8>, i64, memref<?xi8>) -> ()
    %alloc_3 = memref.alloc(%17, %8, %9) : memref<?x?x?xf32>
    %46 = polygeist.memref2pointer %alloc_3 : memref<?x?x?xf32> to !llvm.ptr
    %47 = polygeist.pointer2memref %46 : !llvm.ptr to memref<?xi8>
    %48 = polygeist.pointer2memref %3 : !llvm.ptr to memref<?xi8>
    call @__npb_alloc_check(%47, %14, %48) : (memref<?xi8>, i64, memref<?xi8>) -> ()
    %alloc_4 = memref.alloc(%17, %8, %9) : memref<?x?x?xf32>
    %49 = polygeist.memref2pointer %alloc_4 : memref<?x?x?xf32> to !llvm.ptr
    %50 = polygeist.pointer2memref %49 : !llvm.ptr to memref<?xi8>
    %51 = polygeist.pointer2memref %4 : !llvm.ptr to memref<?xi8>
    call @__npb_alloc_check(%50, %14, %51) : (memref<?xi8>, i64, memref<?xi8>) -> ()
    %alloc_5 = memref.alloc(%29, %22, %9) : memref<?x?x?xf32>
    %52 = polygeist.memref2pointer %alloc_5 : memref<?x?x?xf32> to !llvm.ptr
    %53 = polygeist.pointer2memref %52 : !llvm.ptr to memref<?xi8>
    %54 = polygeist.pointer2memref %5 : !llvm.ptr to memref<?xi8>
    call @__npb_alloc_check(%53, %26, %54) : (memref<?xi8>, i64, memref<?xi8>) -> ()
    %alloc_6 = memref.alloc(%29, %22, %9) : memref<?x?x?xf32>
    %55 = polygeist.memref2pointer %alloc_6 : memref<?x?x?xf32> to !llvm.ptr
    %56 = polygeist.pointer2memref %55 : !llvm.ptr to memref<?xi8>
    %57 = polygeist.pointer2memref %6 : !llvm.ptr to memref<?xi8>
    call @__npb_alloc_check(%56, %26, %57) : (memref<?xi8>, i64, memref<?xi8>) -> ()
    affine.for %arg6 = 0 to #map()[%7] {
      affine.for %arg7 = 0 to #map()[%8] {
        affine.for %arg8 = 0 to %9 {
          %58 = affine.load %arg4[%arg6 + 1, %arg7 + 1, %arg8] : memref<?x?x?xf32>
          %59 = arith.mulf %58, %cst : f32
          %60 = affine.load %arg4[%arg6 + 2, %arg7 + 1, %arg8] : memref<?x?x?xf32>
          %61 = affine.load %arg4[%arg6, %arg7 + 1, %arg8] : memref<?x?x?xf32>
          %62 = arith.addf %60, %61 : f32
          %63 = affine.load %arg4[%arg6 + 1, %arg7 + 2, %arg8] : memref<?x?x?xf32>
          %64 = arith.addf %62, %63 : f32
          %65 = affine.load %arg4[%arg6 + 1, %arg7, %arg8] : memref<?x?x?xf32>
          %66 = arith.addf %64, %65 : f32
          %67 = arith.subf %59, %66 : f32
          affine.store %67, %alloc_2[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
        }
      }
    }
    affine.for %arg6 = 0 to #map1()[%7] {
      affine.for %arg7 = 0 to %8 {
        affine.for %arg8 = 0 to %9 {
          %58 = affine.load %alloc_2[%arg6 + 1, %arg7 + 1, %arg8] : memref<?x?x?xf32>
          %59 = affine.load %alloc_2[%arg6, %arg7 + 1, %arg8] : memref<?x?x?xf32>
          %60 = arith.subf %58, %59 : f32
          affine.store %60, %alloc_3[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
        }
      }
    }
    affine.for %arg6 = 0 to #map1()[%7] {
      affine.for %arg7 = 0 to %8 {
        affine.for %arg8 = 0 to %9 {
          %58 = affine.load %alloc_3[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
          %59 = affine.load %arg4[%arg6 + 2, %arg7 + 2, %arg8] : memref<?x?x?xf32>
          %60 = affine.load %arg4[%arg6 + 1, %arg7 + 2, %arg8] : memref<?x?x?xf32>
          %61 = arith.subf %59, %60 : f32
          %62 = arith.mulf %58, %61 : f32
          %63 = arith.cmpf ogt, %62, %cst_0 : f32
          %64 = arith.select %63, %cst_0, %58 : f32
          affine.store %64, %alloc[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
        }
      }
    }
    affine.for %arg6 = 0 to #map1()[%7] {
      affine.for %arg7 = 0 to %8 {
        affine.for %arg8 = 0 to %9 {
          %58 = affine.load %alloc[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
          affine.store %58, %alloc_4[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
        }
      }
    }
    affine.for %arg6 = 0 to %7 {
      affine.for %arg7 = 0 to #map1()[%8] {
        affine.for %arg8 = 0 to %9 {
          %58 = affine.load %alloc_2[%arg6 + 1, %arg7 + 1, %arg8] : memref<?x?x?xf32>
          %59 = affine.load %alloc_2[%arg6 + 1, %arg7, %arg8] : memref<?x?x?xf32>
          %60 = arith.subf %58, %59 : f32
          affine.store %60, %alloc_5[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
        }
      }
    }
    affine.for %arg6 = 0 to %7 {
      affine.for %arg7 = 0 to #map1()[%8] {
        affine.for %arg8 = 0 to %9 {
          %58 = affine.load %alloc_5[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
          %59 = affine.load %arg4[%arg6 + 2, %arg7 + 2, %arg8] : memref<?x?x?xf32>
          %60 = affine.load %arg4[%arg6 + 2, %arg7 + 1, %arg8] : memref<?x?x?xf32>
          %61 = arith.subf %59, %60 : f32
          %62 = arith.mulf %58, %61 : f32
          %63 = arith.cmpf ogt, %62, %cst_0 : f32
          %64 = arith.select %63, %cst_0, %58 : f32
          affine.store %64, %alloc_1[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
        }
      }
    }
    affine.for %arg6 = 0 to %7 {
      affine.for %arg7 = 0 to #map1()[%8] {
        affine.for %arg8 = 0 to %9 {
          %58 = affine.load %alloc_1[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
          affine.store %58, %alloc_6[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
        }
      }
    }
    affine.for %arg6 = 0 to %7 {
      affine.for %arg7 = 0 to %8 {
        affine.for %arg8 = 0 to %9 {
          %58 = affine.load %arg4[%arg6 + 2, %arg7 + 2, %arg8] : memref<?x?x?xf32>
          %59 = affine.load %arg3[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
          %60 = affine.load %alloc_4[%arg6 + 1, %arg7, %arg8] : memref<?x?x?xf32>
          %61 = affine.load %alloc_4[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
          %62 = arith.subf %60, %61 : f32
          %63 = affine.load %alloc_6[%arg6, %arg7 + 1, %arg8] : memref<?x?x?xf32>
          %64 = arith.addf %62, %63 : f32
          %65 = affine.load %alloc_6[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
          %66 = arith.subf %64, %65 : f32
          %67 = arith.mulf %59, %66 : f32
          %68 = arith.subf %58, %67 : f32
          affine.store %68, %arg5[%arg6, %arg7, %arg8] : memref<?x?x?xf32>
        }
      }
    }
    memref.dealloc %alloc : memref<?x?x?xf32>
    memref.dealloc %alloc_1 : memref<?x?x?xf32>
    memref.dealloc %alloc_2 : memref<?x?x?xf32>
    memref.dealloc %alloc_3 : memref<?x?x?xf32>
    memref.dealloc %alloc_4 : memref<?x?x?xf32>
    memref.dealloc %alloc_5 : memref<?x?x?xf32>
    memref.dealloc %alloc_6 : memref<?x?x?xf32>
    return
  }
  func.func private @__npb_alloc_check(%arg0: memref<?xi8>, %arg1: i64, %arg2: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = llvm.mlir.addressof @str7 : !llvm.ptr
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
