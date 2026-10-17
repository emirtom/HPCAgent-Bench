module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @jacobi_1d_fp64_polygeist(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg7, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg8, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg9, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg10, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg11, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg2, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %arg3, %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg4, %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg5, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg6, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.constant(-3 : index) : i64
    %13 = llvm.mlir.constant(2 : index) : i64
    %14 = llvm.mlir.constant(-2 : index) : i64
    %15 = llvm.mlir.constant(16 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(-1 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    %19 = llvm.mlir.constant(3.333300e-01 : f64) : f64
    llvm.br ^bb1(%18 : i64)
  ^bb1(%20: i64):  // 2 preds: ^bb0, ^bb2
    %21 = llvm.icmp "slt" %20, %arg1 : i64
    llvm.cond_br %21, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %22 = llvm.add %arg0, %17 : i64
    %23 = llvm.add %arg0, %14 : i64
    %24 = llvm.icmp "sle" %23, %16 : i64
    %25 = llvm.sub %13, %arg0 : i64
    %26 = llvm.add %arg0, %12 : i64
    %27 = llvm.select %24, %25, %26 : i1, i64
    %28 = llvm.sdiv %27, %15 : i64
    %29 = llvm.sub %16, %28 : i64
    %30 = llvm.add %28, %18 : i64
    %31 = llvm.select %24, %29, %30 : i1, i64
    %32 = llvm.extractvalue %11[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.extractvalue %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.extractvalue %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %38 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.extractvalue %5[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.extractvalue %5[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    gpu.launch_func  @jacobi_1d_fp64_polygeist_kernel::@jacobi_1d_fp64_polygeist_kernel blocks in (%31, %18, %18) threads in (%15, %18, %18) : i64 args(%15 : i64, %18 : i64, %16 : i64, %22 : i64, %17 : i64, %32 : !llvm.ptr, %33 : !llvm.ptr, %34 : i64, %35 : i64, %36 : i64, %19 : f64, %37 : !llvm.ptr, %38 : !llvm.ptr, %39 : i64, %40 : i64, %41 : i64)
    %42 = llvm.add %arg0, %17 : i64
    %43 = llvm.add %arg0, %14 : i64
    %44 = llvm.icmp "sle" %43, %16 : i64
    %45 = llvm.sub %13, %arg0 : i64
    %46 = llvm.add %arg0, %12 : i64
    %47 = llvm.select %44, %45, %46 : i1, i64
    %48 = llvm.sdiv %47, %15 : i64
    %49 = llvm.sub %16, %48 : i64
    %50 = llvm.add %48, %18 : i64
    %51 = llvm.select %44, %49, %50 : i1, i64
    %52 = llvm.extractvalue %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %53 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.extractvalue %5[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.extractvalue %5[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.extractvalue %11[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %59 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %60 = llvm.extractvalue %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %61 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    gpu.launch_func  @jacobi_1d_fp64_polygeist_kernel_0::@jacobi_1d_fp64_polygeist_kernel blocks in (%51, %18, %18) threads in (%15, %18, %18) : i64 args(%15 : i64, %18 : i64, %16 : i64, %42 : i64, %17 : i64, %52 : !llvm.ptr, %53 : !llvm.ptr, %54 : i64, %55 : i64, %56 : i64, %19 : f64, %57 : !llvm.ptr, %58 : !llvm.ptr, %59 : i64, %60 : i64, %61 : i64)
    %62 = llvm.add %20, %18 : i64
    llvm.br ^bb1(%62 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_jacobi_1d_fp64_polygeist(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @jacobi_1d_fp64_polygeist(%arg0, %arg1, %1, %2, %3, %4, %5, %7, %8, %9, %10, %11) : (i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return
  }
  gpu.module @jacobi_1d_fp64_polygeist_kernel [#rocdl.target<O = 3, chip = "gfx942", features = "+xnack">] attributes {llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"} {
    llvm.func @jacobi_1d_fp64_polygeist_kernel(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: f64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64) attributes {gpu.kernel, rocdl.kernel} {
      %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg11, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %2 = llvm.insertvalue %arg12, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %3 = llvm.insertvalue %arg13, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %4 = llvm.insertvalue %arg14, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %5 = llvm.insertvalue %arg15, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %6 = llvm.insertvalue %arg5, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %7 = llvm.insertvalue %arg6, %6[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %8 = llvm.insertvalue %arg7, %7[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %9 = llvm.insertvalue %arg8, %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %10 = llvm.insertvalue %arg9, %9[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %11 = rocdl.workgroup.id.x : i32
      %12 = llvm.sext %11 : i32 to i64
      %13 = rocdl.workitem.id.x : i32
      %14 = llvm.sext %13 : i32 to i64
      %15 = llvm.mul %12, %arg0 overflow<nsw> : i64
      %16 = llvm.add %15, %arg1 : i64
      %17 = llvm.mul %14, %arg1 overflow<nsw> : i64
      %18 = llvm.add %17, %arg2 : i64
      %19 = llvm.add %18, %16 : i64
      %20 = llvm.mul %18, %arg1 : i64
      %21 = llvm.add %20, %16 : i64
      %22 = llvm.icmp "ult" %21, %arg3 : i64
      llvm.cond_br %22, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %23 = llvm.add %19, %arg4 : i64
      %24 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %25 = llvm.getelementptr inbounds|nuw %24[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %26 = llvm.load %25 : !llvm.ptr -> f64
      %27 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %28 = llvm.getelementptr inbounds|nuw %27[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %29 = llvm.load %28 : !llvm.ptr -> f64
      %30 = llvm.fadd %26, %29 : f64
      %31 = llvm.add %19, %arg1 : i64
      %32 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %33 = llvm.getelementptr inbounds|nuw %32[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %34 = llvm.load %33 : !llvm.ptr -> f64
      %35 = llvm.fadd %30, %34 : f64
      %36 = llvm.fmul %35, %arg10 : f64
      %37 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %38 = llvm.getelementptr inbounds|nuw %37[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %36, %38 : f64, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
  gpu.module @jacobi_1d_fp64_polygeist_kernel_0 [#rocdl.target<O = 3, chip = "gfx942", features = "+xnack">] attributes {llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"} {
    llvm.func @jacobi_1d_fp64_polygeist_kernel(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: f64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64) attributes {gpu.kernel, rocdl.kernel} {
      %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg11, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %2 = llvm.insertvalue %arg12, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %3 = llvm.insertvalue %arg13, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %4 = llvm.insertvalue %arg14, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %5 = llvm.insertvalue %arg15, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %6 = llvm.insertvalue %arg5, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %7 = llvm.insertvalue %arg6, %6[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %8 = llvm.insertvalue %arg7, %7[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %9 = llvm.insertvalue %arg8, %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %10 = llvm.insertvalue %arg9, %9[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %11 = rocdl.workgroup.id.x : i32
      %12 = llvm.sext %11 : i32 to i64
      %13 = rocdl.workitem.id.x : i32
      %14 = llvm.sext %13 : i32 to i64
      %15 = llvm.mul %12, %arg0 overflow<nsw> : i64
      %16 = llvm.add %15, %arg1 : i64
      %17 = llvm.mul %14, %arg1 overflow<nsw> : i64
      %18 = llvm.add %17, %arg2 : i64
      %19 = llvm.add %18, %16 : i64
      %20 = llvm.mul %18, %arg1 : i64
      %21 = llvm.add %20, %16 : i64
      %22 = llvm.icmp "ult" %21, %arg3 : i64
      llvm.cond_br %22, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %23 = llvm.add %19, %arg4 : i64
      %24 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %25 = llvm.getelementptr inbounds|nuw %24[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %26 = llvm.load %25 : !llvm.ptr -> f64
      %27 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %28 = llvm.getelementptr inbounds|nuw %27[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %29 = llvm.load %28 : !llvm.ptr -> f64
      %30 = llvm.fadd %26, %29 : f64
      %31 = llvm.add %19, %arg1 : i64
      %32 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %33 = llvm.getelementptr inbounds|nuw %32[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %34 = llvm.load %33 : !llvm.ptr -> f64
      %35 = llvm.fadd %30, %34 : f64
      %36 = llvm.fmul %35, %arg10 : f64
      %37 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %38 = llvm.getelementptr inbounds|nuw %37[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %36, %38 : f64, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
}

