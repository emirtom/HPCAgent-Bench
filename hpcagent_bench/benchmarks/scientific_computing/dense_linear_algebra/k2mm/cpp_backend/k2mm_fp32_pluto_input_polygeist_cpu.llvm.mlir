module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global internal constant @str3("out of memory: %zu bytes for %s\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str2("__sm1\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("__mm2\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("__mm1\00") {addr_space = 0 : i32}
  llvm.func @k2mm_fp32_polygeist(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: !llvm.ptr, %arg19: !llvm.ptr, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: !llvm.ptr, %arg26: !llvm.ptr, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: f32, %arg33: f32) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(-9223372036854775808 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = llvm.mlir.addressof @str1 : !llvm.ptr
    %4 = llvm.mlir.addressof @str2 : !llvm.ptr
    %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %6 = llvm.mlir.constant(4 : index) : i64
    %7 = llvm.mlir.constant(4 : i64) : i64
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mul %arg0, %arg1 : i64
    %11 = llvm.mul %10, %7 : i64
    %12 = llvm.mul %arg1, %6 : i64
    %13 = llvm.udiv %11, %12 : i64
    %14 = llvm.mul %arg1, %13 : i64
    %15 = llvm.getelementptr %1[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.call @malloc(%16) : (i64) -> !llvm.ptr
    llvm.call @__npb_alloc_check(%17, %17, %8, %0, %9, %11, %2, %2, %8, %0, %9) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %18 = llvm.mul %arg0, %arg3 : i64
    %19 = llvm.mul %18, %7 : i64
    %20 = llvm.mul %arg3, %6 : i64
    %21 = llvm.udiv %19, %20 : i64
    %22 = llvm.mul %arg3, %21 : i64
    %23 = llvm.getelementptr %1[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %24 = llvm.ptrtoint %23 : !llvm.ptr to i64
    %25 = llvm.call @malloc(%24) : (i64) -> !llvm.ptr
    llvm.call @__npb_alloc_check(%25, %25, %8, %0, %9, %19, %3, %3, %8, %0, %9) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %26 = llvm.mul %arg0, %arg2 : i64
    %27 = llvm.mul %26, %7 : i64
    %28 = llvm.mul %arg2, %6 : i64
    %29 = llvm.udiv %27, %28 : i64
    %30 = llvm.mul %arg2, %29 : i64
    %31 = llvm.getelementptr %1[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.call @malloc(%32) : (i64) -> !llvm.ptr
    llvm.call @__npb_alloc_check(%33, %33, %8, %0, %9, %27, %4, %4, %8, %0, %9) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg34) : i64 = (%8) to (%arg0) step (%9) {
          %34 = llvm.intr.stacksave : !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          llvm.br ^bb2(%8 : i64)
        ^bb2(%35: i64):  // 2 preds: ^bb1, ^bb3
          %36 = llvm.icmp "slt" %35, %arg2 : i64
          llvm.cond_br %36, ^bb3, ^bb4
        ^bb3:  // pred: ^bb2
          %37 = llvm.mul %arg34, %arg9 overflow<nsw, nuw> : i64
          %38 = llvm.add %37, %35 overflow<nsw, nuw> : i64
          %39 = llvm.getelementptr inbounds|nuw %arg5[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %40 = llvm.load %39 : !llvm.ptr -> f32
          %41 = llvm.fmul %arg32, %40 : f32
          %42 = llvm.mul %arg34, %arg2 overflow<nsw, nuw> : i64
          %43 = llvm.add %42, %35 overflow<nsw, nuw> : i64
          %44 = llvm.getelementptr inbounds|nuw %33[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %41, %44 : f32, !llvm.ptr
          %45 = llvm.add %35, %9 : i64
          llvm.br ^bb2(%45 : i64)
        ^bb4:  // pred: ^bb2
          llvm.intr.stackrestore %34 : !llvm.ptr
          llvm.br ^bb5
        ^bb5:  // pred: ^bb4
          omp.yield
        }
      }
      omp.terminator
    }
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg34) : i64 = (%8) to (%arg0) step (%9) {
          %34 = llvm.intr.stacksave : !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          llvm.br ^bb2(%8 : i64)
        ^bb2(%35: i64):  // 2 preds: ^bb1, ^bb6
          %36 = llvm.icmp "slt" %35, %arg1 : i64
          llvm.cond_br %36, ^bb3, ^bb7
        ^bb3:  // pred: ^bb2
          %37 = llvm.mul %arg34, %arg1 overflow<nsw, nuw> : i64
          %38 = llvm.add %37, %35 overflow<nsw, nuw> : i64
          %39 = llvm.getelementptr inbounds|nuw %17[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %5, %39 : f32, !llvm.ptr
          llvm.br ^bb4(%8 : i64)
        ^bb4(%40: i64):  // 2 preds: ^bb3, ^bb5
          %41 = llvm.icmp "slt" %40, %arg2 : i64
          llvm.cond_br %41, ^bb5, ^bb6
        ^bb5:  // pred: ^bb4
          %42 = llvm.mul %arg34, %arg2 overflow<nsw, nuw> : i64
          %43 = llvm.add %42, %40 overflow<nsw, nuw> : i64
          %44 = llvm.getelementptr inbounds|nuw %33[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %45 = llvm.load %44 : !llvm.ptr -> f32
          %46 = llvm.mul %40, %arg16 overflow<nsw, nuw> : i64
          %47 = llvm.add %46, %35 overflow<nsw, nuw> : i64
          %48 = llvm.getelementptr inbounds|nuw %arg12[%47] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %49 = llvm.load %48 : !llvm.ptr -> f32
          %50 = llvm.fmul %45, %49 : f32
          %51 = llvm.mul %arg34, %arg1 overflow<nsw, nuw> : i64
          %52 = llvm.add %51, %35 overflow<nsw, nuw> : i64
          %53 = llvm.getelementptr inbounds|nuw %17[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %54 = llvm.load %53 : !llvm.ptr -> f32
          %55 = llvm.fadd %54, %50 : f32
          %56 = llvm.mul %arg34, %arg1 overflow<nsw, nuw> : i64
          %57 = llvm.add %56, %35 overflow<nsw, nuw> : i64
          %58 = llvm.getelementptr inbounds|nuw %17[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %55, %58 : f32, !llvm.ptr
          %59 = llvm.add %40, %9 : i64
          llvm.br ^bb4(%59 : i64)
        ^bb6:  // pred: ^bb4
          %60 = llvm.add %35, %9 : i64
          llvm.br ^bb2(%60 : i64)
        ^bb7:  // pred: ^bb2
          llvm.intr.stackrestore %34 : !llvm.ptr
          llvm.br ^bb8
        ^bb8:  // pred: ^bb7
          omp.yield
        }
      }
      omp.terminator
    }
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg34) : i64 = (%8) to (%arg0) step (%9) {
          %34 = llvm.intr.stacksave : !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          llvm.br ^bb2(%8 : i64)
        ^bb2(%35: i64):  // 2 preds: ^bb1, ^bb6
          %36 = llvm.icmp "slt" %35, %arg3 : i64
          llvm.cond_br %36, ^bb3, ^bb7
        ^bb3:  // pred: ^bb2
          %37 = llvm.mul %arg34, %arg3 overflow<nsw, nuw> : i64
          %38 = llvm.add %37, %35 overflow<nsw, nuw> : i64
          %39 = llvm.getelementptr inbounds|nuw %25[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %5, %39 : f32, !llvm.ptr
          llvm.br ^bb4(%8 : i64)
        ^bb4(%40: i64):  // 2 preds: ^bb3, ^bb5
          %41 = llvm.icmp "slt" %40, %arg1 : i64
          llvm.cond_br %41, ^bb5, ^bb6
        ^bb5:  // pred: ^bb4
          %42 = llvm.mul %arg34, %arg1 overflow<nsw, nuw> : i64
          %43 = llvm.add %42, %40 overflow<nsw, nuw> : i64
          %44 = llvm.getelementptr inbounds|nuw %17[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %45 = llvm.load %44 : !llvm.ptr -> f32
          %46 = llvm.mul %40, %arg23 overflow<nsw, nuw> : i64
          %47 = llvm.add %46, %35 overflow<nsw, nuw> : i64
          %48 = llvm.getelementptr inbounds|nuw %arg19[%47] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %49 = llvm.load %48 : !llvm.ptr -> f32
          %50 = llvm.fmul %45, %49 : f32
          %51 = llvm.mul %arg34, %arg3 overflow<nsw, nuw> : i64
          %52 = llvm.add %51, %35 overflow<nsw, nuw> : i64
          %53 = llvm.getelementptr inbounds|nuw %25[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %54 = llvm.load %53 : !llvm.ptr -> f32
          %55 = llvm.fadd %54, %50 : f32
          %56 = llvm.mul %arg34, %arg3 overflow<nsw, nuw> : i64
          %57 = llvm.add %56, %35 overflow<nsw, nuw> : i64
          %58 = llvm.getelementptr inbounds|nuw %25[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %55, %58 : f32, !llvm.ptr
          %59 = llvm.add %40, %9 : i64
          llvm.br ^bb4(%59 : i64)
        ^bb6:  // pred: ^bb4
          %60 = llvm.add %35, %9 : i64
          llvm.br ^bb2(%60 : i64)
        ^bb7:  // pred: ^bb2
          llvm.intr.stackrestore %34 : !llvm.ptr
          llvm.br ^bb8
        ^bb8:  // pred: ^bb7
          omp.yield
        }
      }
      omp.terminator
    }
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg34) : i64 = (%8) to (%arg0) step (%9) {
          %34 = llvm.intr.stacksave : !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          llvm.br ^bb2(%8 : i64)
        ^bb2(%35: i64):  // 2 preds: ^bb1, ^bb3
          %36 = llvm.icmp "slt" %35, %arg3 : i64
          llvm.cond_br %36, ^bb3, ^bb4
        ^bb3:  // pred: ^bb2
          %37 = llvm.mul %arg34, %arg3 overflow<nsw, nuw> : i64
          %38 = llvm.add %37, %35 overflow<nsw, nuw> : i64
          %39 = llvm.getelementptr inbounds|nuw %25[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %40 = llvm.load %39 : !llvm.ptr -> f32
          %41 = llvm.mul %arg34, %arg30 overflow<nsw, nuw> : i64
          %42 = llvm.add %41, %35 overflow<nsw, nuw> : i64
          %43 = llvm.getelementptr inbounds|nuw %arg26[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %44 = llvm.load %43 : !llvm.ptr -> f32
          %45 = llvm.fmul %arg33, %44 : f32
          %46 = llvm.fadd %40, %45 : f32
          %47 = llvm.mul %arg34, %arg30 overflow<nsw, nuw> : i64
          %48 = llvm.add %47, %35 overflow<nsw, nuw> : i64
          %49 = llvm.getelementptr inbounds|nuw %arg26[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %46, %49 : f32, !llvm.ptr
          %50 = llvm.add %35, %9 : i64
          llvm.br ^bb2(%50 : i64)
        ^bb4:  // pred: ^bb2
          llvm.intr.stackrestore %34 : !llvm.ptr
          llvm.br ^bb5
        ^bb5:  // pred: ^bb4
          omp.yield
        }
      }
      omp.terminator
    }
    llvm.call @free(%17) : (!llvm.ptr) -> ()
    llvm.call @free(%25) : (!llvm.ptr) -> ()
    llvm.call @free(%33) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_k2mm_fp32_polygeist(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: f32, %arg9: f32) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg4 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.load %arg5 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.load %arg6 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.extractvalue %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.extractvalue %16[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.extractvalue %16[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.extractvalue %16[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.extractvalue %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.extractvalue %16[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.load %arg7 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.extractvalue %24[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.extractvalue %24[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.extractvalue %24[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.extractvalue %24[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.extractvalue %24[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.extractvalue %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.extractvalue %24[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @k2mm_fp32_polygeist(%arg0, %arg1, %arg2, %arg3, %1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15, %17, %18, %19, %20, %21, %22, %23, %25, %26, %27, %28, %29, %30, %31, %arg8, %arg9) : (i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, f32, f32) -> ()
    llvm.return
  }
  llvm.func internal @__npb_alloc_check(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: i64, %arg9: i64, %arg10: i64) attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.addressof @stderr : !llvm.ptr
    %3 = llvm.mlir.addressof @str3 : !llvm.ptr
    %4 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %5 = llvm.icmp "eq" %4, %0 : !llvm.ptr
    %6 = llvm.icmp "ne" %arg5, %1 : i64
    %7 = llvm.and %5, %6 : i1
    llvm.cond_br %7, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %8 = llvm.load %2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.getelementptr %10[%9] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>
    %12 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<33 x i8>
    %13 = llvm.getelementptr %arg7[%arg8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %14 = llvm.call @fprintf(%11, %12, %arg5, %13) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
    llvm.call @abort() : () -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return
  }
  llvm.func @abort() attributes {sym_visibility = "private"}
}

