module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @jacobi_1d_fp32_polygeist(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3.333300e-01 : f32) : f32
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(-1 : index) : i64
    llvm.br ^bb1(%1 : i64)
  ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb2
    %4 = llvm.icmp "slt" %3, %arg1 : i64
    llvm.cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %5 = llvm.add %arg0, %2 : i64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg12) : i64 = (%1) to (%5) step (%1) {
          %8 = llvm.intr.stacksave : !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          %9 = llvm.add %arg12, %2 : i64
          %10 = llvm.getelementptr inbounds|nuw %arg3[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %11 = llvm.load %10 : !llvm.ptr -> f32
          %12 = llvm.getelementptr inbounds|nuw %arg3[%arg12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %13 = llvm.load %12 : !llvm.ptr -> f32
          %14 = llvm.fadd %11, %13 : f32
          %15 = llvm.add %arg12, %1 : i64
          %16 = llvm.getelementptr inbounds|nuw %arg3[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %17 = llvm.load %16 : !llvm.ptr -> f32
          %18 = llvm.fadd %14, %17 : f32
          %19 = llvm.fmul %18, %0 : f32
          %20 = llvm.getelementptr inbounds|nuw %arg8[%arg12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %19, %20 : f32, !llvm.ptr
          llvm.intr.stackrestore %8 : !llvm.ptr
          llvm.br ^bb2
        ^bb2:  // pred: ^bb1
          omp.yield
        }
      }
      omp.terminator
    }
    %6 = llvm.add %arg0, %2 : i64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg12) : i64 = (%1) to (%6) step (%1) {
          %8 = llvm.intr.stacksave : !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          %9 = llvm.add %arg12, %2 : i64
          %10 = llvm.getelementptr inbounds|nuw %arg8[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %11 = llvm.load %10 : !llvm.ptr -> f32
          %12 = llvm.getelementptr inbounds|nuw %arg8[%arg12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %13 = llvm.load %12 : !llvm.ptr -> f32
          %14 = llvm.fadd %11, %13 : f32
          %15 = llvm.add %arg12, %1 : i64
          %16 = llvm.getelementptr inbounds|nuw %arg8[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %17 = llvm.load %16 : !llvm.ptr -> f32
          %18 = llvm.fadd %14, %17 : f32
          %19 = llvm.fmul %18, %0 : f32
          %20 = llvm.getelementptr inbounds|nuw %arg3[%arg12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %19, %20 : f32, !llvm.ptr
          llvm.intr.stackrestore %8 : !llvm.ptr
          llvm.br ^bb2
        ^bb2:  // pred: ^bb1
          omp.yield
        }
      }
      omp.terminator
    }
    %7 = llvm.add %3, %1 : i64
    llvm.br ^bb1(%7 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_jacobi_1d_fp32_polygeist(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {llvm.emit_c_interface} {
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
    llvm.call @jacobi_1d_fp32_polygeist(%arg0, %arg1, %1, %2, %3, %4, %5, %7, %8, %9, %10, %11) : (i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return
  }
}

