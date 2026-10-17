module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global internal constant @str7("out of memory: %zu bytes for %s\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str6("fly_field\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("res2\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("flx_field\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("res1\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("lap_field\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("__cb2\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("__cb1\00") {addr_space = 0 : i32}
  llvm.func @hdiff_fp32_polygeist(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: !llvm.ptr, %arg22: !llvm.ptr, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(-9223372036854775808 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(4 : i64) : i64
    %3 = llvm.mlir.constant(4 : index) : i64
    %4 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %5 = llvm.mlir.constant(2 : i64) : i64
    %6 = llvm.mlir.constant(4.000000e+00 : f32) : f32
    %7 = llvm.mlir.addressof @str6 : !llvm.ptr
    %8 = llvm.mlir.addressof @str5 : !llvm.ptr
    %9 = llvm.mlir.addressof @str4 : !llvm.ptr
    %10 = llvm.mlir.addressof @str3 : !llvm.ptr
    %11 = llvm.mlir.addressof @str2 : !llvm.ptr
    %12 = llvm.mlir.addressof @str1 : !llvm.ptr
    %13 = llvm.mlir.addressof @str0 : !llvm.ptr
    %14 = llvm.mlir.constant(1 : i64) : i64
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.mlir.constant(2 : index) : i64
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.mul %arg1, %arg2 : i64
    %19 = llvm.add %arg0, %14 : i64
    %20 = llvm.mul %19, %arg1 : i64
    %21 = llvm.mul %20, %arg2 : i64
    %22 = llvm.mul %21, %2 : i64
    %23 = llvm.mul %18, %3 : i64
    %24 = llvm.udiv %22, %23 : i64
    %25 = llvm.mul %arg2, %arg1 : i64
    %26 = llvm.mul %25, %24 : i64
    %27 = llvm.getelementptr %1[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.call @malloc(%28) : (i64) -> !llvm.ptr
    llvm.call @__npb_alloc_check(%29, %29, %15, %0, %17, %22, %13, %13, %15, %0, %17) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %30 = llvm.add %arg1, %14 : i64
    %31 = llvm.mul %30, %arg2 : i64
    %32 = llvm.mul %arg0, %30 : i64
    %33 = llvm.mul %32, %arg2 : i64
    %34 = llvm.mul %33, %2 : i64
    %35 = llvm.mul %31, %3 : i64
    %36 = llvm.udiv %34, %35 : i64
    %37 = llvm.mul %arg2, %30 : i64
    %38 = llvm.mul %37, %36 : i64
    %39 = llvm.getelementptr %1[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.call @malloc(%40) : (i64) -> !llvm.ptr
    llvm.call @__npb_alloc_check(%41, %41, %15, %0, %17, %34, %12, %12, %15, %0, %17) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %42 = llvm.add %arg1, %5 : i64
    %43 = llvm.mul %42, %arg2 : i64
    %44 = llvm.add %arg0, %5 : i64
    %45 = llvm.mul %44, %42 : i64
    %46 = llvm.mul %45, %arg2 : i64
    %47 = llvm.mul %46, %2 : i64
    %48 = llvm.mul %43, %3 : i64
    %49 = llvm.udiv %47, %48 : i64
    %50 = llvm.mul %arg2, %42 : i64
    %51 = llvm.mul %50, %49 : i64
    %52 = llvm.getelementptr %1[%51] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %53 = llvm.ptrtoint %52 : !llvm.ptr to i64
    %54 = llvm.call @malloc(%53) : (i64) -> !llvm.ptr
    llvm.call @__npb_alloc_check(%54, %54, %15, %0, %17, %47, %11, %11, %15, %0, %17) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %55 = llvm.mul %arg2, %arg1 : i64
    %56 = llvm.mul %55, %24 : i64
    %57 = llvm.getelementptr %1[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %58 = llvm.ptrtoint %57 : !llvm.ptr to i64
    %59 = llvm.call @malloc(%58) : (i64) -> !llvm.ptr
    llvm.call @__npb_alloc_check(%59, %59, %15, %0, %17, %22, %10, %10, %15, %0, %17) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %60 = llvm.mul %arg2, %arg1 : i64
    %61 = llvm.mul %60, %24 : i64
    %62 = llvm.getelementptr %1[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.call @malloc(%63) : (i64) -> !llvm.ptr
    llvm.call @__npb_alloc_check(%64, %64, %15, %0, %17, %22, %9, %9, %15, %0, %17) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %65 = llvm.mul %arg2, %30 : i64
    %66 = llvm.mul %65, %36 : i64
    %67 = llvm.getelementptr %1[%66] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %68 = llvm.ptrtoint %67 : !llvm.ptr to i64
    %69 = llvm.call @malloc(%68) : (i64) -> !llvm.ptr
    llvm.call @__npb_alloc_check(%69, %69, %15, %0, %17, %34, %8, %8, %15, %0, %17) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %70 = llvm.mul %arg2, %30 : i64
    %71 = llvm.mul %70, %36 : i64
    %72 = llvm.getelementptr %1[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %73 = llvm.ptrtoint %72 : !llvm.ptr to i64
    %74 = llvm.call @malloc(%73) : (i64) -> !llvm.ptr
    llvm.call @__npb_alloc_check(%74, %74, %15, %0, %17, %34, %7, %7, %15, %0, %17) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %75 = llvm.add %arg0, %16 : i64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg30) : i64 = (%15) to (%75) step (%17) {
          %79 = llvm.intr.stacksave : !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          %80 = llvm.add %arg1, %16 : i64
          llvm.br ^bb2(%15 : i64)
        ^bb2(%81: i64):  // 2 preds: ^bb1, ^bb6
          %82 = llvm.icmp "slt" %81, %80 : i64
          llvm.cond_br %82, ^bb3, ^bb7
        ^bb3:  // pred: ^bb2
          llvm.br ^bb4(%15 : i64)
        ^bb4(%83: i64):  // 2 preds: ^bb3, ^bb5
          %84 = llvm.icmp "slt" %83, %arg2 : i64
          llvm.cond_br %84, ^bb5, ^bb6
        ^bb5:  // pred: ^bb4
          %85 = llvm.add %arg30, %17 : i64
          %86 = llvm.add %81, %17 : i64
          %87 = llvm.mul %85, %arg18 overflow<nsw, nuw> : i64
          %88 = llvm.mul %86, %arg19 overflow<nsw, nuw> : i64
          %89 = llvm.add %87, %88 overflow<nsw, nuw> : i64
          %90 = llvm.add %89, %83 overflow<nsw, nuw> : i64
          %91 = llvm.getelementptr inbounds|nuw %arg13[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %92 = llvm.load %91 : !llvm.ptr -> f32
          %93 = llvm.fmul %92, %6 : f32
          %94 = llvm.add %arg30, %16 : i64
          %95 = llvm.add %81, %17 : i64
          %96 = llvm.mul %94, %arg18 overflow<nsw, nuw> : i64
          %97 = llvm.mul %95, %arg19 overflow<nsw, nuw> : i64
          %98 = llvm.add %96, %97 overflow<nsw, nuw> : i64
          %99 = llvm.add %98, %83 overflow<nsw, nuw> : i64
          %100 = llvm.getelementptr inbounds|nuw %arg13[%99] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %101 = llvm.load %100 : !llvm.ptr -> f32
          %102 = llvm.add %81, %17 : i64
          %103 = llvm.mul %arg30, %arg18 overflow<nsw, nuw> : i64
          %104 = llvm.mul %102, %arg19 overflow<nsw, nuw> : i64
          %105 = llvm.add %103, %104 overflow<nsw, nuw> : i64
          %106 = llvm.add %105, %83 overflow<nsw, nuw> : i64
          %107 = llvm.getelementptr inbounds|nuw %arg13[%106] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %108 = llvm.load %107 : !llvm.ptr -> f32
          %109 = llvm.fadd %101, %108 : f32
          %110 = llvm.add %arg30, %17 : i64
          %111 = llvm.add %81, %16 : i64
          %112 = llvm.mul %110, %arg18 overflow<nsw, nuw> : i64
          %113 = llvm.mul %111, %arg19 overflow<nsw, nuw> : i64
          %114 = llvm.add %112, %113 overflow<nsw, nuw> : i64
          %115 = llvm.add %114, %83 overflow<nsw, nuw> : i64
          %116 = llvm.getelementptr inbounds|nuw %arg13[%115] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %117 = llvm.load %116 : !llvm.ptr -> f32
          %118 = llvm.fadd %109, %117 : f32
          %119 = llvm.add %arg30, %17 : i64
          %120 = llvm.mul %119, %arg18 overflow<nsw, nuw> : i64
          %121 = llvm.mul %81, %arg19 overflow<nsw, nuw> : i64
          %122 = llvm.add %120, %121 overflow<nsw, nuw> : i64
          %123 = llvm.add %122, %83 overflow<nsw, nuw> : i64
          %124 = llvm.getelementptr inbounds|nuw %arg13[%123] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %125 = llvm.load %124 : !llvm.ptr -> f32
          %126 = llvm.fadd %118, %125 : f32
          %127 = llvm.fsub %93, %126 : f32
          %128 = llvm.mul %arg30, %50 overflow<nsw, nuw> : i64
          %129 = llvm.mul %81, %arg2 overflow<nsw, nuw> : i64
          %130 = llvm.add %128, %129 overflow<nsw, nuw> : i64
          %131 = llvm.add %130, %83 overflow<nsw, nuw> : i64
          %132 = llvm.getelementptr inbounds|nuw %54[%131] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %127, %132 : f32, !llvm.ptr
          %133 = llvm.add %83, %17 : i64
          llvm.br ^bb4(%133 : i64)
        ^bb6:  // pred: ^bb4
          %134 = llvm.add %81, %17 : i64
          llvm.br ^bb2(%134 : i64)
        ^bb7:  // pred: ^bb2
          llvm.intr.stackrestore %79 : !llvm.ptr
          llvm.br ^bb8
        ^bb8:  // pred: ^bb7
          omp.yield
        }
      }
      omp.terminator
    }
    %76 = llvm.add %arg0, %17 : i64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg30) : i64 = (%15) to (%76) step (%17) {
          %79 = llvm.intr.stacksave : !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          llvm.br ^bb2(%15 : i64)
        ^bb2(%80: i64):  // 2 preds: ^bb1, ^bb6
          %81 = llvm.icmp "slt" %80, %arg1 : i64
          llvm.cond_br %81, ^bb3, ^bb7
        ^bb3:  // pred: ^bb2
          llvm.br ^bb4(%15 : i64)
        ^bb4(%82: i64):  // 2 preds: ^bb3, ^bb5
          %83 = llvm.icmp "slt" %82, %arg2 : i64
          llvm.cond_br %83, ^bb5, ^bb6
        ^bb5:  // pred: ^bb4
          %84 = llvm.add %arg30, %17 : i64
          %85 = llvm.add %80, %17 : i64
          %86 = llvm.mul %84, %50 overflow<nsw, nuw> : i64
          %87 = llvm.mul %85, %arg2 overflow<nsw, nuw> : i64
          %88 = llvm.add %86, %87 overflow<nsw, nuw> : i64
          %89 = llvm.add %88, %82 overflow<nsw, nuw> : i64
          %90 = llvm.getelementptr inbounds|nuw %54[%89] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %91 = llvm.load %90 : !llvm.ptr -> f32
          %92 = llvm.add %80, %17 : i64
          %93 = llvm.mul %arg30, %50 overflow<nsw, nuw> : i64
          %94 = llvm.mul %92, %arg2 overflow<nsw, nuw> : i64
          %95 = llvm.add %93, %94 overflow<nsw, nuw> : i64
          %96 = llvm.add %95, %82 overflow<nsw, nuw> : i64
          %97 = llvm.getelementptr inbounds|nuw %54[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %98 = llvm.load %97 : !llvm.ptr -> f32
          %99 = llvm.fsub %91, %98 : f32
          %100 = llvm.mul %arg30, %55 overflow<nsw, nuw> : i64
          %101 = llvm.mul %80, %arg2 overflow<nsw, nuw> : i64
          %102 = llvm.add %100, %101 overflow<nsw, nuw> : i64
          %103 = llvm.add %102, %82 overflow<nsw, nuw> : i64
          %104 = llvm.getelementptr inbounds|nuw %59[%103] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %99, %104 : f32, !llvm.ptr
          %105 = llvm.add %82, %17 : i64
          llvm.br ^bb4(%105 : i64)
        ^bb6:  // pred: ^bb4
          %106 = llvm.add %80, %17 : i64
          llvm.br ^bb2(%106 : i64)
        ^bb7:  // pred: ^bb2
          llvm.intr.stackrestore %79 : !llvm.ptr
          llvm.br ^bb8
        ^bb8:  // pred: ^bb7
          omp.yield
        }
      }
      omp.terminator
    }
    %77 = llvm.add %arg0, %17 : i64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg30) : i64 = (%15) to (%77) step (%17) {
          %79 = llvm.intr.stacksave : !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          llvm.br ^bb2(%15 : i64)
        ^bb2(%80: i64):  // 2 preds: ^bb1, ^bb6
          %81 = llvm.icmp "slt" %80, %arg1 : i64
          llvm.cond_br %81, ^bb3, ^bb7
        ^bb3:  // pred: ^bb2
          llvm.br ^bb4(%15 : i64)
        ^bb4(%82: i64):  // 2 preds: ^bb3, ^bb5
          %83 = llvm.icmp "slt" %82, %arg2 : i64
          llvm.cond_br %83, ^bb5, ^bb6
        ^bb5:  // pred: ^bb4
          %84 = llvm.mul %arg30, %55 overflow<nsw, nuw> : i64
          %85 = llvm.mul %80, %arg2 overflow<nsw, nuw> : i64
          %86 = llvm.add %84, %85 overflow<nsw, nuw> : i64
          %87 = llvm.add %86, %82 overflow<nsw, nuw> : i64
          %88 = llvm.getelementptr inbounds|nuw %59[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %89 = llvm.load %88 : !llvm.ptr -> f32
          %90 = llvm.add %arg30, %16 : i64
          %91 = llvm.add %80, %16 : i64
          %92 = llvm.mul %90, %arg18 overflow<nsw, nuw> : i64
          %93 = llvm.mul %91, %arg19 overflow<nsw, nuw> : i64
          %94 = llvm.add %92, %93 overflow<nsw, nuw> : i64
          %95 = llvm.add %94, %82 overflow<nsw, nuw> : i64
          %96 = llvm.getelementptr inbounds|nuw %arg13[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %97 = llvm.load %96 : !llvm.ptr -> f32
          %98 = llvm.add %arg30, %17 : i64
          %99 = llvm.add %80, %16 : i64
          %100 = llvm.mul %98, %arg18 overflow<nsw, nuw> : i64
          %101 = llvm.mul %99, %arg19 overflow<nsw, nuw> : i64
          %102 = llvm.add %100, %101 overflow<nsw, nuw> : i64
          %103 = llvm.add %102, %82 overflow<nsw, nuw> : i64
          %104 = llvm.getelementptr inbounds|nuw %arg13[%103] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %105 = llvm.load %104 : !llvm.ptr -> f32
          %106 = llvm.fsub %97, %105 : f32
          %107 = llvm.fmul %89, %106 : f32
          %108 = llvm.fcmp "ogt" %107, %4 : f32
          %109 = llvm.select %108, %4, %89 : i1, f32
          %110 = llvm.mul %arg30, %25 overflow<nsw, nuw> : i64
          %111 = llvm.mul %80, %arg2 overflow<nsw, nuw> : i64
          %112 = llvm.add %110, %111 overflow<nsw, nuw> : i64
          %113 = llvm.add %112, %82 overflow<nsw, nuw> : i64
          %114 = llvm.getelementptr inbounds|nuw %29[%113] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %109, %114 : f32, !llvm.ptr
          %115 = llvm.add %82, %17 : i64
          llvm.br ^bb4(%115 : i64)
        ^bb6:  // pred: ^bb4
          %116 = llvm.add %80, %17 : i64
          llvm.br ^bb2(%116 : i64)
        ^bb7:  // pred: ^bb2
          llvm.intr.stackrestore %79 : !llvm.ptr
          llvm.br ^bb8
        ^bb8:  // pred: ^bb7
          omp.yield
        }
      }
      omp.terminator
    }
    %78 = llvm.add %arg0, %17 : i64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg30) : i64 = (%15) to (%78) step (%17) {
          %79 = llvm.intr.stacksave : !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          llvm.br ^bb2(%15 : i64)
        ^bb2(%80: i64):  // 2 preds: ^bb1, ^bb6
          %81 = llvm.icmp "slt" %80, %arg1 : i64
          llvm.cond_br %81, ^bb3, ^bb7
        ^bb3:  // pred: ^bb2
          llvm.br ^bb4(%15 : i64)
        ^bb4(%82: i64):  // 2 preds: ^bb3, ^bb5
          %83 = llvm.icmp "slt" %82, %arg2 : i64
          llvm.cond_br %83, ^bb5, ^bb6
        ^bb5:  // pred: ^bb4
          %84 = llvm.mul %arg30, %25 overflow<nsw, nuw> : i64
          %85 = llvm.mul %80, %arg2 overflow<nsw, nuw> : i64
          %86 = llvm.add %84, %85 overflow<nsw, nuw> : i64
          %87 = llvm.add %86, %82 overflow<nsw, nuw> : i64
          %88 = llvm.getelementptr inbounds|nuw %29[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %89 = llvm.load %88 : !llvm.ptr -> f32
          %90 = llvm.mul %arg30, %60 overflow<nsw, nuw> : i64
          %91 = llvm.mul %80, %arg2 overflow<nsw, nuw> : i64
          %92 = llvm.add %90, %91 overflow<nsw, nuw> : i64
          %93 = llvm.add %92, %82 overflow<nsw, nuw> : i64
          %94 = llvm.getelementptr inbounds|nuw %64[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %89, %94 : f32, !llvm.ptr
          %95 = llvm.add %82, %17 : i64
          llvm.br ^bb4(%95 : i64)
        ^bb6:  // pred: ^bb4
          %96 = llvm.add %80, %17 : i64
          llvm.br ^bb2(%96 : i64)
        ^bb7:  // pred: ^bb2
          llvm.intr.stackrestore %79 : !llvm.ptr
          llvm.br ^bb8
        ^bb8:  // pred: ^bb7
          omp.yield
        }
      }
      omp.terminator
    }
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg30) : i64 = (%15) to (%arg0) step (%17) {
          %79 = llvm.intr.stacksave : !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          %80 = llvm.add %arg1, %17 : i64
          llvm.br ^bb2(%15 : i64)
        ^bb2(%81: i64):  // 2 preds: ^bb1, ^bb6
          %82 = llvm.icmp "slt" %81, %80 : i64
          llvm.cond_br %82, ^bb3, ^bb7
        ^bb3:  // pred: ^bb2
          llvm.br ^bb4(%15 : i64)
        ^bb4(%83: i64):  // 2 preds: ^bb3, ^bb5
          %84 = llvm.icmp "slt" %83, %arg2 : i64
          llvm.cond_br %84, ^bb5, ^bb6
        ^bb5:  // pred: ^bb4
          %85 = llvm.add %arg30, %17 : i64
          %86 = llvm.add %81, %17 : i64
          %87 = llvm.mul %85, %50 overflow<nsw, nuw> : i64
          %88 = llvm.mul %86, %arg2 overflow<nsw, nuw> : i64
          %89 = llvm.add %87, %88 overflow<nsw, nuw> : i64
          %90 = llvm.add %89, %83 overflow<nsw, nuw> : i64
          %91 = llvm.getelementptr inbounds|nuw %54[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %92 = llvm.load %91 : !llvm.ptr -> f32
          %93 = llvm.add %arg30, %17 : i64
          %94 = llvm.mul %93, %50 overflow<nsw, nuw> : i64
          %95 = llvm.mul %81, %arg2 overflow<nsw, nuw> : i64
          %96 = llvm.add %94, %95 overflow<nsw, nuw> : i64
          %97 = llvm.add %96, %83 overflow<nsw, nuw> : i64
          %98 = llvm.getelementptr inbounds|nuw %54[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %99 = llvm.load %98 : !llvm.ptr -> f32
          %100 = llvm.fsub %92, %99 : f32
          %101 = llvm.mul %arg30, %65 overflow<nsw, nuw> : i64
          %102 = llvm.mul %81, %arg2 overflow<nsw, nuw> : i64
          %103 = llvm.add %101, %102 overflow<nsw, nuw> : i64
          %104 = llvm.add %103, %83 overflow<nsw, nuw> : i64
          %105 = llvm.getelementptr inbounds|nuw %69[%104] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %100, %105 : f32, !llvm.ptr
          %106 = llvm.add %83, %17 : i64
          llvm.br ^bb4(%106 : i64)
        ^bb6:  // pred: ^bb4
          %107 = llvm.add %81, %17 : i64
          llvm.br ^bb2(%107 : i64)
        ^bb7:  // pred: ^bb2
          llvm.intr.stackrestore %79 : !llvm.ptr
          llvm.br ^bb8
        ^bb8:  // pred: ^bb7
          omp.yield
        }
      }
      omp.terminator
    }
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg30) : i64 = (%15) to (%arg0) step (%17) {
          %79 = llvm.intr.stacksave : !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          %80 = llvm.add %arg1, %17 : i64
          llvm.br ^bb2(%15 : i64)
        ^bb2(%81: i64):  // 2 preds: ^bb1, ^bb6
          %82 = llvm.icmp "slt" %81, %80 : i64
          llvm.cond_br %82, ^bb3, ^bb7
        ^bb3:  // pred: ^bb2
          llvm.br ^bb4(%15 : i64)
        ^bb4(%83: i64):  // 2 preds: ^bb3, ^bb5
          %84 = llvm.icmp "slt" %83, %arg2 : i64
          llvm.cond_br %84, ^bb5, ^bb6
        ^bb5:  // pred: ^bb4
          %85 = llvm.mul %arg30, %65 overflow<nsw, nuw> : i64
          %86 = llvm.mul %81, %arg2 overflow<nsw, nuw> : i64
          %87 = llvm.add %85, %86 overflow<nsw, nuw> : i64
          %88 = llvm.add %87, %83 overflow<nsw, nuw> : i64
          %89 = llvm.getelementptr inbounds|nuw %69[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %90 = llvm.load %89 : !llvm.ptr -> f32
          %91 = llvm.add %arg30, %16 : i64
          %92 = llvm.add %81, %16 : i64
          %93 = llvm.mul %91, %arg18 overflow<nsw, nuw> : i64
          %94 = llvm.mul %92, %arg19 overflow<nsw, nuw> : i64
          %95 = llvm.add %93, %94 overflow<nsw, nuw> : i64
          %96 = llvm.add %95, %83 overflow<nsw, nuw> : i64
          %97 = llvm.getelementptr inbounds|nuw %arg13[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %98 = llvm.load %97 : !llvm.ptr -> f32
          %99 = llvm.add %arg30, %16 : i64
          %100 = llvm.add %81, %17 : i64
          %101 = llvm.mul %99, %arg18 overflow<nsw, nuw> : i64
          %102 = llvm.mul %100, %arg19 overflow<nsw, nuw> : i64
          %103 = llvm.add %101, %102 overflow<nsw, nuw> : i64
          %104 = llvm.add %103, %83 overflow<nsw, nuw> : i64
          %105 = llvm.getelementptr inbounds|nuw %arg13[%104] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %106 = llvm.load %105 : !llvm.ptr -> f32
          %107 = llvm.fsub %98, %106 : f32
          %108 = llvm.fmul %90, %107 : f32
          %109 = llvm.fcmp "ogt" %108, %4 : f32
          %110 = llvm.select %109, %4, %90 : i1, f32
          %111 = llvm.mul %arg30, %37 overflow<nsw, nuw> : i64
          %112 = llvm.mul %81, %arg2 overflow<nsw, nuw> : i64
          %113 = llvm.add %111, %112 overflow<nsw, nuw> : i64
          %114 = llvm.add %113, %83 overflow<nsw, nuw> : i64
          %115 = llvm.getelementptr inbounds|nuw %41[%114] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %110, %115 : f32, !llvm.ptr
          %116 = llvm.add %83, %17 : i64
          llvm.br ^bb4(%116 : i64)
        ^bb6:  // pred: ^bb4
          %117 = llvm.add %81, %17 : i64
          llvm.br ^bb2(%117 : i64)
        ^bb7:  // pred: ^bb2
          llvm.intr.stackrestore %79 : !llvm.ptr
          llvm.br ^bb8
        ^bb8:  // pred: ^bb7
          omp.yield
        }
      }
      omp.terminator
    }
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg30) : i64 = (%15) to (%arg0) step (%17) {
          %79 = llvm.intr.stacksave : !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          %80 = llvm.add %arg1, %17 : i64
          llvm.br ^bb2(%15 : i64)
        ^bb2(%81: i64):  // 2 preds: ^bb1, ^bb6
          %82 = llvm.icmp "slt" %81, %80 : i64
          llvm.cond_br %82, ^bb3, ^bb7
        ^bb3:  // pred: ^bb2
          llvm.br ^bb4(%15 : i64)
        ^bb4(%83: i64):  // 2 preds: ^bb3, ^bb5
          %84 = llvm.icmp "slt" %83, %arg2 : i64
          llvm.cond_br %84, ^bb5, ^bb6
        ^bb5:  // pred: ^bb4
          %85 = llvm.mul %arg30, %37 overflow<nsw, nuw> : i64
          %86 = llvm.mul %81, %arg2 overflow<nsw, nuw> : i64
          %87 = llvm.add %85, %86 overflow<nsw, nuw> : i64
          %88 = llvm.add %87, %83 overflow<nsw, nuw> : i64
          %89 = llvm.getelementptr inbounds|nuw %41[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %90 = llvm.load %89 : !llvm.ptr -> f32
          %91 = llvm.mul %arg30, %70 overflow<nsw, nuw> : i64
          %92 = llvm.mul %81, %arg2 overflow<nsw, nuw> : i64
          %93 = llvm.add %91, %92 overflow<nsw, nuw> : i64
          %94 = llvm.add %93, %83 overflow<nsw, nuw> : i64
          %95 = llvm.getelementptr inbounds|nuw %74[%94] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %90, %95 : f32, !llvm.ptr
          %96 = llvm.add %83, %17 : i64
          llvm.br ^bb4(%96 : i64)
        ^bb6:  // pred: ^bb4
          %97 = llvm.add %81, %17 : i64
          llvm.br ^bb2(%97 : i64)
        ^bb7:  // pred: ^bb2
          llvm.intr.stackrestore %79 : !llvm.ptr
          llvm.br ^bb8
        ^bb8:  // pred: ^bb7
          omp.yield
        }
      }
      omp.terminator
    }
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg30) : i64 = (%15) to (%arg0) step (%17) {
          %79 = llvm.intr.stacksave : !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          llvm.br ^bb2(%15 : i64)
        ^bb2(%80: i64):  // 2 preds: ^bb1, ^bb6
          %81 = llvm.icmp "slt" %80, %arg1 : i64
          llvm.cond_br %81, ^bb3, ^bb7
        ^bb3:  // pred: ^bb2
          llvm.br ^bb4(%15 : i64)
        ^bb4(%82: i64):  // 2 preds: ^bb3, ^bb5
          %83 = llvm.icmp "slt" %82, %arg2 : i64
          llvm.cond_br %83, ^bb5, ^bb6
        ^bb5:  // pred: ^bb4
          %84 = llvm.add %arg30, %16 : i64
          %85 = llvm.add %80, %16 : i64
          %86 = llvm.mul %84, %arg18 overflow<nsw, nuw> : i64
          %87 = llvm.mul %85, %arg19 overflow<nsw, nuw> : i64
          %88 = llvm.add %86, %87 overflow<nsw, nuw> : i64
          %89 = llvm.add %88, %82 overflow<nsw, nuw> : i64
          %90 = llvm.getelementptr inbounds|nuw %arg13[%89] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %91 = llvm.load %90 : !llvm.ptr -> f32
          %92 = llvm.mul %arg30, %arg9 overflow<nsw, nuw> : i64
          %93 = llvm.mul %80, %arg10 overflow<nsw, nuw> : i64
          %94 = llvm.add %92, %93 overflow<nsw, nuw> : i64
          %95 = llvm.add %94, %82 overflow<nsw, nuw> : i64
          %96 = llvm.getelementptr inbounds|nuw %arg4[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %97 = llvm.load %96 : !llvm.ptr -> f32
          %98 = llvm.add %arg30, %17 : i64
          %99 = llvm.mul %98, %60 overflow<nsw, nuw> : i64
          %100 = llvm.mul %80, %arg2 overflow<nsw, nuw> : i64
          %101 = llvm.add %99, %100 overflow<nsw, nuw> : i64
          %102 = llvm.add %101, %82 overflow<nsw, nuw> : i64
          %103 = llvm.getelementptr inbounds|nuw %64[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %104 = llvm.load %103 : !llvm.ptr -> f32
          %105 = llvm.mul %arg30, %60 overflow<nsw, nuw> : i64
          %106 = llvm.mul %80, %arg2 overflow<nsw, nuw> : i64
          %107 = llvm.add %105, %106 overflow<nsw, nuw> : i64
          %108 = llvm.add %107, %82 overflow<nsw, nuw> : i64
          %109 = llvm.getelementptr inbounds|nuw %64[%108] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %110 = llvm.load %109 : !llvm.ptr -> f32
          %111 = llvm.fsub %104, %110 : f32
          %112 = llvm.add %80, %17 : i64
          %113 = llvm.mul %arg30, %70 overflow<nsw, nuw> : i64
          %114 = llvm.mul %112, %arg2 overflow<nsw, nuw> : i64
          %115 = llvm.add %113, %114 overflow<nsw, nuw> : i64
          %116 = llvm.add %115, %82 overflow<nsw, nuw> : i64
          %117 = llvm.getelementptr inbounds|nuw %74[%116] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %118 = llvm.load %117 : !llvm.ptr -> f32
          %119 = llvm.fadd %111, %118 : f32
          %120 = llvm.mul %arg30, %70 overflow<nsw, nuw> : i64
          %121 = llvm.mul %80, %arg2 overflow<nsw, nuw> : i64
          %122 = llvm.add %120, %121 overflow<nsw, nuw> : i64
          %123 = llvm.add %122, %82 overflow<nsw, nuw> : i64
          %124 = llvm.getelementptr inbounds|nuw %74[%123] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          %125 = llvm.load %124 : !llvm.ptr -> f32
          %126 = llvm.fsub %119, %125 : f32
          %127 = llvm.fmul %97, %126 : f32
          %128 = llvm.fsub %91, %127 : f32
          %129 = llvm.mul %arg30, %arg27 overflow<nsw, nuw> : i64
          %130 = llvm.mul %80, %arg28 overflow<nsw, nuw> : i64
          %131 = llvm.add %129, %130 overflow<nsw, nuw> : i64
          %132 = llvm.add %131, %82 overflow<nsw, nuw> : i64
          %133 = llvm.getelementptr inbounds|nuw %arg22[%132] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %128, %133 : f32, !llvm.ptr
          %134 = llvm.add %82, %17 : i64
          llvm.br ^bb4(%134 : i64)
        ^bb6:  // pred: ^bb4
          %135 = llvm.add %80, %17 : i64
          llvm.br ^bb2(%135 : i64)
        ^bb7:  // pred: ^bb2
          llvm.intr.stackrestore %79 : !llvm.ptr
          llvm.br ^bb8
        ^bb8:  // pred: ^bb7
          omp.yield
        }
      }
      omp.terminator
    }
    llvm.call @free(%29) : (!llvm.ptr) -> ()
    llvm.call @free(%41) : (!llvm.ptr) -> ()
    llvm.call @free(%54) : (!llvm.ptr) -> ()
    llvm.call @free(%59) : (!llvm.ptr) -> ()
    llvm.call @free(%64) : (!llvm.ptr) -> ()
    llvm.call @free(%69) : (!llvm.ptr) -> ()
    llvm.call @free(%74) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_hdiff_fp32_polygeist(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %8 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.load %arg4 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %12 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.extractvalue %10[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.extractvalue %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.extractvalue %10[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.extractvalue %10[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.extractvalue %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.extractvalue %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.extractvalue %10[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.load %arg5 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %21 = llvm.extractvalue %20[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.extractvalue %20[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %23 = llvm.extractvalue %20[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.extractvalue %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.extractvalue %20[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.extractvalue %20[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %27 = llvm.extractvalue %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %28 = llvm.extractvalue %20[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %29 = llvm.extractvalue %20[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.call @hdiff_fp32_polygeist(%arg0, %arg1, %arg2, %1, %2, %3, %4, %5, %6, %7, %8, %9, %11, %12, %13, %14, %15, %16, %17, %18, %19, %21, %22, %23, %24, %25, %26, %27, %28, %29) : (i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> ()
    llvm.return
  }
  llvm.func internal @__npb_alloc_check(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: i64, %arg9: i64, %arg10: i64) attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.addressof @stderr : !llvm.ptr
    %3 = llvm.mlir.addressof @str7 : !llvm.ptr
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

