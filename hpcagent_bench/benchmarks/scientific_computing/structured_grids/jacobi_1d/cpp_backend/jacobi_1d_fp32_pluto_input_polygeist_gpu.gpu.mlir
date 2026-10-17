module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d_fp32_polygeist(%arg0: i64, %arg1: i64, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.emit_c_interface, llvm.linkage = #llvm.linkage<external>} {
    %c-3 = arith.constant -3 : index
    %c2 = arith.constant 2 : index
    %c-2 = arith.constant -2 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c-1 = arith.constant -1 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 3.333300e-01 : f32
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.index_cast %arg1 : i64 to index
    scf.for %arg4 = %c1 to %1 step %c1 {
      %2 = arith.addi %0, %c-1 : index
      %3 = arith.addi %0, %c-2 : index
      %4 = arith.cmpi sle, %3, %c0 : index
      %5 = arith.subi %c2, %0 : index
      %6 = arith.addi %0, %c-3 : index
      %7 = arith.select %4, %5, %6 : index
      %8 = arith.divsi %7, %c16 : index
      %9 = arith.subi %c0, %8 : index
      %10 = arith.addi %8, %c1 : index
      %11 = arith.select %4, %9, %10 : index
      gpu.launch_func  @jacobi_1d_fp32_polygeist_kernel::@jacobi_1d_fp32_polygeist_kernel blocks in (%11, %c1, %c1) threads in (%c16, %c1, %c1)  args(%c16 : index, %c1 : index, %c0 : index, %2 : index, %c-1 : index, %arg2 : memref<?xf32>, %cst : f32, %arg3 : memref<?xf32>)
      %12 = arith.addi %0, %c-1 : index
      %13 = arith.addi %0, %c-2 : index
      %14 = arith.cmpi sle, %13, %c0 : index
      %15 = arith.subi %c2, %0 : index
      %16 = arith.addi %0, %c-3 : index
      %17 = arith.select %14, %15, %16 : index
      %18 = arith.divsi %17, %c16 : index
      %19 = arith.subi %c0, %18 : index
      %20 = arith.addi %18, %c1 : index
      %21 = arith.select %14, %19, %20 : index
      gpu.launch_func  @jacobi_1d_fp32_polygeist_kernel_0::@jacobi_1d_fp32_polygeist_kernel blocks in (%21, %c1, %c1) threads in (%c16, %c1, %c1)  args(%c16 : index, %c1 : index, %c0 : index, %12 : index, %c-1 : index, %arg3 : memref<?xf32>, %cst : f32, %arg2 : memref<?xf32>)
    }
    return
  }
  gpu.module @jacobi_1d_fp32_polygeist_kernel {
    gpu.func @jacobi_1d_fp32_polygeist_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: memref<?xf32>, %arg6: f32, %arg7: memref<?xf32>) kernel {
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_id_x, %arg0 overflow<nsw> : index
      %1 = arith.addi %0, %arg1 : index
      %2 = arith.muli %thread_id_x, %arg1 overflow<nsw> : index
      %3 = arith.addi %2, %arg2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.muli %3, %arg1 : index
      %6 = arith.addi %5, %1 : index
      %7 = arith.cmpi ult, %6, %arg3 : index
      scf.if %7 {
        %8 = arith.addi %4, %arg4 : index
        %9 = memref.load %arg5[%8] : memref<?xf32>
        %10 = memref.load %arg5[%4] : memref<?xf32>
        %11 = arith.addf %9, %10 : f32
        %12 = arith.addi %4, %arg1 : index
        %13 = memref.load %arg5[%12] : memref<?xf32>
        %14 = arith.addf %11, %13 : f32
        %15 = arith.mulf %14, %arg6 : f32
        memref.store %15, %arg7[%4] : memref<?xf32>
      }
      gpu.return
    }
  }
  gpu.module @jacobi_1d_fp32_polygeist_kernel_0 {
    gpu.func @jacobi_1d_fp32_polygeist_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: memref<?xf32>, %arg6: f32, %arg7: memref<?xf32>) kernel {
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_id_x, %arg0 overflow<nsw> : index
      %1 = arith.addi %0, %arg1 : index
      %2 = arith.muli %thread_id_x, %arg1 overflow<nsw> : index
      %3 = arith.addi %2, %arg2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.muli %3, %arg1 : index
      %6 = arith.addi %5, %1 : index
      %7 = arith.cmpi ult, %6, %arg3 : index
      scf.if %7 {
        %8 = arith.addi %4, %arg4 : index
        %9 = memref.load %arg5[%8] : memref<?xf32>
        %10 = memref.load %arg5[%4] : memref<?xf32>
        %11 = arith.addf %9, %10 : f32
        %12 = arith.addi %4, %arg1 : index
        %13 = memref.load %arg5[%12] : memref<?xf32>
        %14 = arith.addf %11, %13 : f32
        %15 = arith.mulf %14, %arg6 : f32
        memref.store %15, %arg7[%4] : memref<?xf32>
      }
      gpu.return
    }
  }
}

