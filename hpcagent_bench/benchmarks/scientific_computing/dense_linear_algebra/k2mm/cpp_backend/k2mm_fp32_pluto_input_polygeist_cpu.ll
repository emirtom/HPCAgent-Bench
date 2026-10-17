; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@str3 = internal constant [33 x i8] c"out of memory: %zu bytes for %s\0A\00"
@stderr = external global { ptr, ptr, i64, [1 x i64], [1 x i64] }
@str2 = internal constant [6 x i8] c"__sm1\00"
@str1 = internal constant [6 x i8] c"__mm2\00"
@str0 = internal constant [6 x i8] c"__mm1\00"
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 22, ptr @0 }, align 8

declare void @free(ptr)

declare ptr @malloc(i64)

declare i32 @fprintf(ptr, ptr, ...)

define void @k2mm_fp32_polygeist(i64 %0, i64 %1, i64 %2, i64 %3, ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, ptr %18, ptr %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, ptr %25, ptr %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, float %32, float %33) {
  %structArg170 = alloca { ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %structArg165 = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %structArg160 = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %structArg = alloca { ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %.reloaded154 = alloca i64, align 8
  %.reloaded155 = alloca i64, align 8
  %.reloaded156 = alloca i64, align 8
  %.reloaded157 = alloca float, align 4
  %.reloaded108 = alloca i64, align 8
  %.reloaded109 = alloca i64, align 8
  %.reloaded110 = alloca i64, align 8
  %.reloaded111 = alloca i64, align 8
  %.reloaded59 = alloca i64, align 8
  %.reloaded60 = alloca i64, align 8
  %.reloaded61 = alloca i64, align 8
  %.reloaded62 = alloca i64, align 8
  %.reloaded = alloca i64, align 8
  %.reloaded12 = alloca i64, align 8
  %.reloaded13 = alloca i64, align 8
  %.reloaded14 = alloca float, align 4
  %35 = mul i64 %0, %1
  %36 = mul i64 %35, 4
  %37 = mul i64 %1, 4
  %38 = udiv i64 %36, %37
  %39 = mul i64 %1, %38
  %40 = getelementptr float, ptr null, i64 %39
  %41 = ptrtoint ptr %40 to i64
  %42 = call ptr @malloc(i64 %41)
  call void @__npb_alloc_check(ptr %42, ptr %42, i64 0, i64 -9223372036854775808, i64 1, i64 %36, ptr @str0, ptr @str0, i64 0, i64 -9223372036854775808, i64 1)
  %43 = mul i64 %0, %3
  %44 = mul i64 %43, 4
  %45 = mul i64 %3, 4
  %46 = udiv i64 %44, %45
  %47 = mul i64 %3, %46
  %48 = getelementptr float, ptr null, i64 %47
  %49 = ptrtoint ptr %48 to i64
  %50 = call ptr @malloc(i64 %49)
  call void @__npb_alloc_check(ptr %50, ptr %50, i64 0, i64 -9223372036854775808, i64 1, i64 %44, ptr @str1, ptr @str1, i64 0, i64 -9223372036854775808, i64 1)
  %51 = mul i64 %0, %2
  %52 = mul i64 %51, 4
  %53 = mul i64 %2, 4
  %54 = udiv i64 %52, %53
  %55 = mul i64 %2, %54
  %56 = getelementptr float, ptr null, i64 %55
  %57 = ptrtoint ptr %56 to i64
  %58 = call ptr @malloc(i64 %57)
  call void @__npb_alloc_check(ptr %58, ptr %58, i64 0, i64 -9223372036854775808, i64 1, i64 %52, ptr @str2, ptr @str2, i64 0, i64 -9223372036854775808, i64 1)
  br label %entry

entry:                                            ; preds = %34
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %0, ptr %.reloaded, align 8
  store i64 %2, ptr %.reloaded12, align 8
  store i64 %9, ptr %.reloaded13, align 8
  store float %32, ptr %.reloaded14, align 4
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_.reloaded = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 0
  store ptr %.reloaded, ptr %gep_.reloaded, align 8
  %gep_.reloaded12 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 1
  store ptr %.reloaded12, ptr %gep_.reloaded12, align 8
  %gep_.reloaded13 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 2
  store ptr %.reloaded13, ptr %gep_.reloaded13, align 8
  %gep_.reloaded14 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 3
  store ptr %.reloaded14, ptr %gep_.reloaded14, align 8
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 4
  store ptr %5, ptr %gep_, align 8
  %gep_159 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 5
  store ptr %58, ptr %gep_159, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @k2mm_fp32_polygeist..omp_par, ptr %structArg)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  %omp_global_thread_num15 = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %0, ptr %.reloaded59, align 8
  store i64 %1, ptr %.reloaded60, align 8
  store i64 %2, ptr %.reloaded61, align 8
  store i64 %16, ptr %.reloaded62, align 8
  br label %omp_parallel164

omp_parallel164:                                  ; preds = %omp.par.exit
  %gep_.reloaded59 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg160, i32 0, i32 0
  store ptr %.reloaded59, ptr %gep_.reloaded59, align 8
  %gep_.reloaded60 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg160, i32 0, i32 1
  store ptr %.reloaded60, ptr %gep_.reloaded60, align 8
  %gep_.reloaded61 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg160, i32 0, i32 2
  store ptr %.reloaded61, ptr %gep_.reloaded61, align 8
  %gep_.reloaded62 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg160, i32 0, i32 3
  store ptr %.reloaded62, ptr %gep_.reloaded62, align 8
  %gep_161 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg160, i32 0, i32 4
  store ptr %42, ptr %gep_161, align 8
  %gep_162 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg160, i32 0, i32 5
  store ptr %58, ptr %gep_162, align 8
  %gep_163 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg160, i32 0, i32 6
  store ptr %12, ptr %gep_163, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @k2mm_fp32_polygeist..omp_par.1, ptr %structArg160)
  br label %omp.par.exit21

omp.par.exit21:                                   ; preds = %omp_parallel164
  %omp_global_thread_num64 = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %0, ptr %.reloaded108, align 8
  store i64 %3, ptr %.reloaded109, align 8
  store i64 %1, ptr %.reloaded110, align 8
  store i64 %23, ptr %.reloaded111, align 8
  br label %omp_parallel169

omp_parallel169:                                  ; preds = %omp.par.exit21
  %gep_.reloaded108 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg165, i32 0, i32 0
  store ptr %.reloaded108, ptr %gep_.reloaded108, align 8
  %gep_.reloaded109 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg165, i32 0, i32 1
  store ptr %.reloaded109, ptr %gep_.reloaded109, align 8
  %gep_.reloaded110 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg165, i32 0, i32 2
  store ptr %.reloaded110, ptr %gep_.reloaded110, align 8
  %gep_.reloaded111 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg165, i32 0, i32 3
  store ptr %.reloaded111, ptr %gep_.reloaded111, align 8
  %gep_166 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg165, i32 0, i32 4
  store ptr %50, ptr %gep_166, align 8
  %gep_167 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg165, i32 0, i32 5
  store ptr %42, ptr %gep_167, align 8
  %gep_168 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg165, i32 0, i32 6
  store ptr %19, ptr %gep_168, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @k2mm_fp32_polygeist..omp_par.2, ptr %structArg165)
  br label %omp.par.exit70

omp.par.exit70:                                   ; preds = %omp_parallel169
  %omp_global_thread_num113 = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %0, ptr %.reloaded154, align 8
  store i64 %3, ptr %.reloaded155, align 8
  store i64 %30, ptr %.reloaded156, align 8
  store float %33, ptr %.reloaded157, align 4
  br label %omp_parallel173

omp_parallel173:                                  ; preds = %omp.par.exit70
  %gep_.reloaded154 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg170, i32 0, i32 0
  store ptr %.reloaded154, ptr %gep_.reloaded154, align 8
  %gep_.reloaded155 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg170, i32 0, i32 1
  store ptr %.reloaded155, ptr %gep_.reloaded155, align 8
  %gep_.reloaded156 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg170, i32 0, i32 2
  store ptr %.reloaded156, ptr %gep_.reloaded156, align 8
  %gep_.reloaded157 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg170, i32 0, i32 3
  store ptr %.reloaded157, ptr %gep_.reloaded157, align 8
  %gep_171 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg170, i32 0, i32 4
  store ptr %50, ptr %gep_171, align 8
  %gep_172 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg170, i32 0, i32 5
  store ptr %26, ptr %gep_172, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @k2mm_fp32_polygeist..omp_par.3, ptr %structArg170)
  br label %omp.par.exit119

omp.par.exit119:                                  ; preds = %omp_parallel173
  call void @free(ptr %42)
  call void @free(ptr %50)
  call void @free(ptr %58)
  ret void
}

; Function Attrs: nounwind
define internal void @k2mm_fp32_polygeist..omp_par.3(ptr noalias %tid.addr114, ptr noalias %zero.addr115, ptr %0) #0 {
omp.par.entry116:
  %gep_.reloaded154 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded154 = load ptr, ptr %gep_.reloaded154, align 8, !align !1
  %gep_.reloaded155 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded155 = load ptr, ptr %gep_.reloaded155, align 8, !align !1
  %gep_.reloaded156 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded156 = load ptr, ptr %gep_.reloaded156, align 8, !align !1
  %gep_.reloaded157 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded157 = load ptr, ptr %gep_.reloaded157, align 8, !align !2
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !3
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 5
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !3
  %p.lastiter148 = alloca i32, align 4
  %p.lowerbound149 = alloca i64, align 8
  %p.upperbound150 = alloca i64, align 8
  %p.stride151 = alloca i64, align 8
  %tid.addr.local120 = alloca i32, align 4
  %1 = load i32, ptr %tid.addr114, align 4
  store i32 %1, ptr %tid.addr.local120, align 4
  %tid121 = load i32, ptr %tid.addr.local120, align 4
  %2 = load i64, ptr %loadgep_.reloaded154, align 8
  %3 = load i64, ptr %loadgep_.reloaded155, align 8
  %4 = load i64, ptr %loadgep_.reloaded156, align 8
  %5 = load float, ptr %loadgep_.reloaded157, align 4
  br label %omp.region.after_alloca127

omp.region.after_alloca127:                       ; preds = %omp.par.entry116
  br label %omp.region.after_alloca124

omp.region.after_alloca124:                       ; preds = %omp.region.after_alloca127
  br label %omp.par.region117

omp.par.region117:                                ; preds = %omp.region.after_alloca124
  br label %omp.par.region126

omp.par.region126:                                ; preds = %omp.par.region117
  br label %omp.wsloop.region129

omp.wsloop.region129:                             ; preds = %omp.par.region126
  %6 = select i1 false, i64 %2, i64 0
  %7 = select i1 false, i64 0, i64 %2
  %8 = sub nsw i64 %7, %6
  %9 = icmp sle i64 %7, %6
  %10 = sub i64 %8, 1
  %11 = udiv i64 %10, 1
  %12 = add i64 %11, 1
  %13 = icmp ule i64 %8, 1
  %14 = select i1 %13, i64 1, i64 %12
  %omp_loop.tripcount130 = select i1 %9, i64 0, i64 %14
  br label %omp_loop.preheader131

omp_loop.preheader131:                            ; preds = %omp.wsloop.region129
  store i64 0, ptr %p.lowerbound149, align 8
  %15 = sub i64 %omp_loop.tripcount130, 1
  store i64 %15, ptr %p.upperbound150, align 8
  store i64 1, ptr %p.stride151, align 8
  %omp_global_thread_num152 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num152, i32 34, ptr %p.lastiter148, ptr %p.lowerbound149, ptr %p.upperbound150, ptr %p.stride151, i64 1, i64 0)
  %16 = load i64, ptr %p.lowerbound149, align 8
  %17 = load i64, ptr %p.upperbound150, align 8
  %18 = sub i64 %17, %16
  %19 = add i64 %18, 1
  br label %omp_loop.header132

omp_loop.header132:                               ; preds = %omp_loop.inc135, %omp_loop.preheader131
  %omp_loop.iv138 = phi i64 [ 0, %omp_loop.preheader131 ], [ %omp_loop.next140, %omp_loop.inc135 ]
  br label %omp_loop.cond133

omp_loop.cond133:                                 ; preds = %omp_loop.header132
  %omp_loop.cmp139 = icmp ult i64 %omp_loop.iv138, %19
  br i1 %omp_loop.cmp139, label %omp_loop.body134, label %omp_loop.exit136

omp_loop.exit136:                                 ; preds = %omp_loop.cond133
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num152)
  %omp_global_thread_num153 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num153)
  br label %omp_loop.after137

omp_loop.after137:                                ; preds = %omp_loop.exit136
  br label %omp.region.cont128

omp.region.cont128:                               ; preds = %omp_loop.after137
  br label %omp.region.cont125

omp.region.cont125:                               ; preds = %omp.region.cont128
  br label %omp.par.pre_finalize118

omp.par.pre_finalize118:                          ; preds = %omp.region.cont125
  br label %.fini158

.fini158:                                         ; preds = %omp.par.pre_finalize118
  br label %omp.par.exit119.exitStub

omp_loop.body134:                                 ; preds = %omp_loop.cond133
  %20 = add i64 %omp_loop.iv138, %16
  %21 = mul i64 %20, 1
  %22 = add i64 %21, 0
  br label %omp.loop_nest.region142

omp.loop_nest.region142:                          ; preds = %omp_loop.body134
  %23 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region143

omp.loop_nest.region143:                          ; preds = %omp.loop_nest.region142
  br label %omp.loop_nest.region144

omp.loop_nest.region144:                          ; preds = %omp.loop_nest.region145, %omp.loop_nest.region143
  %24 = phi i64 [ %39, %omp.loop_nest.region145 ], [ 0, %omp.loop_nest.region143 ]
  %25 = icmp slt i64 %24, %3
  br i1 %25, label %omp.loop_nest.region145, label %omp.loop_nest.region146

omp.loop_nest.region146:                          ; preds = %omp.loop_nest.region144
  call void @llvm.stackrestore.p0(ptr %23)
  br label %omp.loop_nest.region147

omp.loop_nest.region147:                          ; preds = %omp.loop_nest.region146
  br label %omp.region.cont141

omp.region.cont141:                               ; preds = %omp.loop_nest.region147
  br label %omp_loop.inc135

omp_loop.inc135:                                  ; preds = %omp.region.cont141
  %omp_loop.next140 = add nuw i64 %omp_loop.iv138, 1
  br label %omp_loop.header132

omp.loop_nest.region145:                          ; preds = %omp.loop_nest.region144
  %26 = mul nuw nsw i64 %22, %3
  %27 = add nuw nsw i64 %26, %24
  %28 = getelementptr inbounds nuw float, ptr %loadgep_, i64 %27
  %29 = load float, ptr %28, align 4
  %30 = mul nuw nsw i64 %22, %4
  %31 = add nuw nsw i64 %30, %24
  %32 = getelementptr inbounds nuw float, ptr %loadgep_2, i64 %31
  %33 = load float, ptr %32, align 4
  %34 = fmul float %5, %33
  %35 = fadd float %29, %34
  %36 = mul nuw nsw i64 %22, %4
  %37 = add nuw nsw i64 %36, %24
  %38 = getelementptr inbounds nuw float, ptr %loadgep_2, i64 %37
  store float %35, ptr %38, align 4
  %39 = add i64 %24, 1
  br label %omp.loop_nest.region144

omp.par.exit119.exitStub:                         ; preds = %.fini158
  ret void
}

; Function Attrs: nounwind
define internal void @k2mm_fp32_polygeist..omp_par.2(ptr noalias %tid.addr65, ptr noalias %zero.addr66, ptr %0) #0 {
omp.par.entry67:
  %gep_.reloaded108 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded108 = load ptr, ptr %gep_.reloaded108, align 8, !align !1
  %gep_.reloaded109 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded109 = load ptr, ptr %gep_.reloaded109, align 8, !align !1
  %gep_.reloaded110 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded110 = load ptr, ptr %gep_.reloaded110, align 8, !align !1
  %gep_.reloaded111 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded111 = load ptr, ptr %gep_.reloaded111, align 8, !align !1
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !3
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 5
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !3
  %gep_3 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 6
  %loadgep_4 = load ptr, ptr %gep_3, align 8, !align !3
  %p.lastiter102 = alloca i32, align 4
  %p.lowerbound103 = alloca i64, align 8
  %p.upperbound104 = alloca i64, align 8
  %p.stride105 = alloca i64, align 8
  %tid.addr.local71 = alloca i32, align 4
  %1 = load i32, ptr %tid.addr65, align 4
  store i32 %1, ptr %tid.addr.local71, align 4
  %tid72 = load i32, ptr %tid.addr.local71, align 4
  %2 = load i64, ptr %loadgep_.reloaded108, align 8
  %3 = load i64, ptr %loadgep_.reloaded109, align 8
  %4 = load i64, ptr %loadgep_.reloaded110, align 8
  %5 = load i64, ptr %loadgep_.reloaded111, align 8
  br label %omp.region.after_alloca78

omp.region.after_alloca78:                        ; preds = %omp.par.entry67
  br label %omp.region.after_alloca75

omp.region.after_alloca75:                        ; preds = %omp.region.after_alloca78
  br label %omp.par.region68

omp.par.region68:                                 ; preds = %omp.region.after_alloca75
  br label %omp.par.region77

omp.par.region77:                                 ; preds = %omp.par.region68
  br label %omp.wsloop.region80

omp.wsloop.region80:                              ; preds = %omp.par.region77
  %6 = select i1 false, i64 %2, i64 0
  %7 = select i1 false, i64 0, i64 %2
  %8 = sub nsw i64 %7, %6
  %9 = icmp sle i64 %7, %6
  %10 = sub i64 %8, 1
  %11 = udiv i64 %10, 1
  %12 = add i64 %11, 1
  %13 = icmp ule i64 %8, 1
  %14 = select i1 %13, i64 1, i64 %12
  %omp_loop.tripcount81 = select i1 %9, i64 0, i64 %14
  br label %omp_loop.preheader82

omp_loop.preheader82:                             ; preds = %omp.wsloop.region80
  store i64 0, ptr %p.lowerbound103, align 8
  %15 = sub i64 %omp_loop.tripcount81, 1
  store i64 %15, ptr %p.upperbound104, align 8
  store i64 1, ptr %p.stride105, align 8
  %omp_global_thread_num106 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num106, i32 34, ptr %p.lastiter102, ptr %p.lowerbound103, ptr %p.upperbound104, ptr %p.stride105, i64 1, i64 0)
  %16 = load i64, ptr %p.lowerbound103, align 8
  %17 = load i64, ptr %p.upperbound104, align 8
  %18 = sub i64 %17, %16
  %19 = add i64 %18, 1
  br label %omp_loop.header83

omp_loop.header83:                                ; preds = %omp_loop.inc86, %omp_loop.preheader82
  %omp_loop.iv89 = phi i64 [ 0, %omp_loop.preheader82 ], [ %omp_loop.next91, %omp_loop.inc86 ]
  br label %omp_loop.cond84

omp_loop.cond84:                                  ; preds = %omp_loop.header83
  %omp_loop.cmp90 = icmp ult i64 %omp_loop.iv89, %19
  br i1 %omp_loop.cmp90, label %omp_loop.body85, label %omp_loop.exit87

omp_loop.exit87:                                  ; preds = %omp_loop.cond84
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num106)
  %omp_global_thread_num107 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num107)
  br label %omp_loop.after88

omp_loop.after88:                                 ; preds = %omp_loop.exit87
  br label %omp.region.cont79

omp.region.cont79:                                ; preds = %omp_loop.after88
  br label %omp.region.cont76

omp.region.cont76:                                ; preds = %omp.region.cont79
  br label %omp.par.pre_finalize69

omp.par.pre_finalize69:                           ; preds = %omp.region.cont76
  br label %.fini112

.fini112:                                         ; preds = %omp.par.pre_finalize69
  br label %omp.par.exit70.exitStub

omp_loop.body85:                                  ; preds = %omp_loop.cond84
  %20 = add i64 %omp_loop.iv89, %16
  %21 = mul i64 %20, 1
  %22 = add i64 %21, 0
  br label %omp.loop_nest.region93

omp.loop_nest.region93:                           ; preds = %omp_loop.body85
  %23 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region94

omp.loop_nest.region94:                           ; preds = %omp.loop_nest.region93
  br label %omp.loop_nest.region95

omp.loop_nest.region95:                           ; preds = %omp.loop_nest.region99, %omp.loop_nest.region94
  %24 = phi i64 [ %31, %omp.loop_nest.region99 ], [ 0, %omp.loop_nest.region94 ]
  %25 = icmp slt i64 %24, %3
  br i1 %25, label %omp.loop_nest.region96, label %omp.loop_nest.region100

omp.loop_nest.region100:                          ; preds = %omp.loop_nest.region95
  call void @llvm.stackrestore.p0(ptr %23)
  br label %omp.loop_nest.region101

omp.loop_nest.region101:                          ; preds = %omp.loop_nest.region100
  br label %omp.region.cont92

omp.region.cont92:                                ; preds = %omp.loop_nest.region101
  br label %omp_loop.inc86

omp_loop.inc86:                                   ; preds = %omp.region.cont92
  %omp_loop.next91 = add nuw i64 %omp_loop.iv89, 1
  br label %omp_loop.header83

omp.loop_nest.region96:                           ; preds = %omp.loop_nest.region95
  %26 = mul nuw nsw i64 %22, %3
  %27 = add nuw nsw i64 %26, %24
  %28 = getelementptr inbounds nuw float, ptr %loadgep_, i64 %27
  store float 0.000000e+00, ptr %28, align 4
  br label %omp.loop_nest.region97

omp.loop_nest.region97:                           ; preds = %omp.loop_nest.region98, %omp.loop_nest.region96
  %29 = phi i64 [ %49, %omp.loop_nest.region98 ], [ 0, %omp.loop_nest.region96 ]
  %30 = icmp slt i64 %29, %4
  br i1 %30, label %omp.loop_nest.region98, label %omp.loop_nest.region99

omp.loop_nest.region99:                           ; preds = %omp.loop_nest.region97
  %31 = add i64 %24, 1
  br label %omp.loop_nest.region95

omp.loop_nest.region98:                           ; preds = %omp.loop_nest.region97
  %32 = mul nuw nsw i64 %22, %4
  %33 = add nuw nsw i64 %32, %29
  %34 = getelementptr inbounds nuw float, ptr %loadgep_2, i64 %33
  %35 = load float, ptr %34, align 4
  %36 = mul nuw nsw i64 %29, %5
  %37 = add nuw nsw i64 %36, %24
  %38 = getelementptr inbounds nuw float, ptr %loadgep_4, i64 %37
  %39 = load float, ptr %38, align 4
  %40 = fmul float %35, %39
  %41 = mul nuw nsw i64 %22, %3
  %42 = add nuw nsw i64 %41, %24
  %43 = getelementptr inbounds nuw float, ptr %loadgep_, i64 %42
  %44 = load float, ptr %43, align 4
  %45 = fadd float %44, %40
  %46 = mul nuw nsw i64 %22, %3
  %47 = add nuw nsw i64 %46, %24
  %48 = getelementptr inbounds nuw float, ptr %loadgep_, i64 %47
  store float %45, ptr %48, align 4
  %49 = add i64 %29, 1
  br label %omp.loop_nest.region97

omp.par.exit70.exitStub:                          ; preds = %.fini112
  ret void
}

; Function Attrs: nounwind
define internal void @k2mm_fp32_polygeist..omp_par.1(ptr noalias %tid.addr16, ptr noalias %zero.addr17, ptr %0) #0 {
omp.par.entry18:
  %gep_.reloaded59 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded59 = load ptr, ptr %gep_.reloaded59, align 8, !align !1
  %gep_.reloaded60 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded60 = load ptr, ptr %gep_.reloaded60, align 8, !align !1
  %gep_.reloaded61 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded61 = load ptr, ptr %gep_.reloaded61, align 8, !align !1
  %gep_.reloaded62 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded62 = load ptr, ptr %gep_.reloaded62, align 8, !align !1
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !3
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 5
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !3
  %gep_3 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 6
  %loadgep_4 = load ptr, ptr %gep_3, align 8, !align !3
  %p.lastiter53 = alloca i32, align 4
  %p.lowerbound54 = alloca i64, align 8
  %p.upperbound55 = alloca i64, align 8
  %p.stride56 = alloca i64, align 8
  %tid.addr.local22 = alloca i32, align 4
  %1 = load i32, ptr %tid.addr16, align 4
  store i32 %1, ptr %tid.addr.local22, align 4
  %tid23 = load i32, ptr %tid.addr.local22, align 4
  %2 = load i64, ptr %loadgep_.reloaded59, align 8
  %3 = load i64, ptr %loadgep_.reloaded60, align 8
  %4 = load i64, ptr %loadgep_.reloaded61, align 8
  %5 = load i64, ptr %loadgep_.reloaded62, align 8
  br label %omp.region.after_alloca29

omp.region.after_alloca29:                        ; preds = %omp.par.entry18
  br label %omp.region.after_alloca26

omp.region.after_alloca26:                        ; preds = %omp.region.after_alloca29
  br label %omp.par.region19

omp.par.region19:                                 ; preds = %omp.region.after_alloca26
  br label %omp.par.region28

omp.par.region28:                                 ; preds = %omp.par.region19
  br label %omp.wsloop.region31

omp.wsloop.region31:                              ; preds = %omp.par.region28
  %6 = select i1 false, i64 %2, i64 0
  %7 = select i1 false, i64 0, i64 %2
  %8 = sub nsw i64 %7, %6
  %9 = icmp sle i64 %7, %6
  %10 = sub i64 %8, 1
  %11 = udiv i64 %10, 1
  %12 = add i64 %11, 1
  %13 = icmp ule i64 %8, 1
  %14 = select i1 %13, i64 1, i64 %12
  %omp_loop.tripcount32 = select i1 %9, i64 0, i64 %14
  br label %omp_loop.preheader33

omp_loop.preheader33:                             ; preds = %omp.wsloop.region31
  store i64 0, ptr %p.lowerbound54, align 8
  %15 = sub i64 %omp_loop.tripcount32, 1
  store i64 %15, ptr %p.upperbound55, align 8
  store i64 1, ptr %p.stride56, align 8
  %omp_global_thread_num57 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num57, i32 34, ptr %p.lastiter53, ptr %p.lowerbound54, ptr %p.upperbound55, ptr %p.stride56, i64 1, i64 0)
  %16 = load i64, ptr %p.lowerbound54, align 8
  %17 = load i64, ptr %p.upperbound55, align 8
  %18 = sub i64 %17, %16
  %19 = add i64 %18, 1
  br label %omp_loop.header34

omp_loop.header34:                                ; preds = %omp_loop.inc37, %omp_loop.preheader33
  %omp_loop.iv40 = phi i64 [ 0, %omp_loop.preheader33 ], [ %omp_loop.next42, %omp_loop.inc37 ]
  br label %omp_loop.cond35

omp_loop.cond35:                                  ; preds = %omp_loop.header34
  %omp_loop.cmp41 = icmp ult i64 %omp_loop.iv40, %19
  br i1 %omp_loop.cmp41, label %omp_loop.body36, label %omp_loop.exit38

omp_loop.exit38:                                  ; preds = %omp_loop.cond35
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num57)
  %omp_global_thread_num58 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num58)
  br label %omp_loop.after39

omp_loop.after39:                                 ; preds = %omp_loop.exit38
  br label %omp.region.cont30

omp.region.cont30:                                ; preds = %omp_loop.after39
  br label %omp.region.cont27

omp.region.cont27:                                ; preds = %omp.region.cont30
  br label %omp.par.pre_finalize20

omp.par.pre_finalize20:                           ; preds = %omp.region.cont27
  br label %.fini63

.fini63:                                          ; preds = %omp.par.pre_finalize20
  br label %omp.par.exit21.exitStub

omp_loop.body36:                                  ; preds = %omp_loop.cond35
  %20 = add i64 %omp_loop.iv40, %16
  %21 = mul i64 %20, 1
  %22 = add i64 %21, 0
  br label %omp.loop_nest.region44

omp.loop_nest.region44:                           ; preds = %omp_loop.body36
  %23 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region45

omp.loop_nest.region45:                           ; preds = %omp.loop_nest.region44
  br label %omp.loop_nest.region46

omp.loop_nest.region46:                           ; preds = %omp.loop_nest.region50, %omp.loop_nest.region45
  %24 = phi i64 [ %31, %omp.loop_nest.region50 ], [ 0, %omp.loop_nest.region45 ]
  %25 = icmp slt i64 %24, %3
  br i1 %25, label %omp.loop_nest.region47, label %omp.loop_nest.region51

omp.loop_nest.region51:                           ; preds = %omp.loop_nest.region46
  call void @llvm.stackrestore.p0(ptr %23)
  br label %omp.loop_nest.region52

omp.loop_nest.region52:                           ; preds = %omp.loop_nest.region51
  br label %omp.region.cont43

omp.region.cont43:                                ; preds = %omp.loop_nest.region52
  br label %omp_loop.inc37

omp_loop.inc37:                                   ; preds = %omp.region.cont43
  %omp_loop.next42 = add nuw i64 %omp_loop.iv40, 1
  br label %omp_loop.header34

omp.loop_nest.region47:                           ; preds = %omp.loop_nest.region46
  %26 = mul nuw nsw i64 %22, %3
  %27 = add nuw nsw i64 %26, %24
  %28 = getelementptr inbounds nuw float, ptr %loadgep_, i64 %27
  store float 0.000000e+00, ptr %28, align 4
  br label %omp.loop_nest.region48

omp.loop_nest.region48:                           ; preds = %omp.loop_nest.region49, %omp.loop_nest.region47
  %29 = phi i64 [ %49, %omp.loop_nest.region49 ], [ 0, %omp.loop_nest.region47 ]
  %30 = icmp slt i64 %29, %4
  br i1 %30, label %omp.loop_nest.region49, label %omp.loop_nest.region50

omp.loop_nest.region50:                           ; preds = %omp.loop_nest.region48
  %31 = add i64 %24, 1
  br label %omp.loop_nest.region46

omp.loop_nest.region49:                           ; preds = %omp.loop_nest.region48
  %32 = mul nuw nsw i64 %22, %4
  %33 = add nuw nsw i64 %32, %29
  %34 = getelementptr inbounds nuw float, ptr %loadgep_2, i64 %33
  %35 = load float, ptr %34, align 4
  %36 = mul nuw nsw i64 %29, %5
  %37 = add nuw nsw i64 %36, %24
  %38 = getelementptr inbounds nuw float, ptr %loadgep_4, i64 %37
  %39 = load float, ptr %38, align 4
  %40 = fmul float %35, %39
  %41 = mul nuw nsw i64 %22, %3
  %42 = add nuw nsw i64 %41, %24
  %43 = getelementptr inbounds nuw float, ptr %loadgep_, i64 %42
  %44 = load float, ptr %43, align 4
  %45 = fadd float %44, %40
  %46 = mul nuw nsw i64 %22, %3
  %47 = add nuw nsw i64 %46, %24
  %48 = getelementptr inbounds nuw float, ptr %loadgep_, i64 %47
  store float %45, ptr %48, align 4
  %49 = add i64 %29, 1
  br label %omp.loop_nest.region48

omp.par.exit21.exitStub:                          ; preds = %.fini63
  ret void
}

; Function Attrs: nounwind
define internal void @k2mm_fp32_polygeist..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #0 {
omp.par.entry:
  %gep_.reloaded = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded = load ptr, ptr %gep_.reloaded, align 8, !align !1
  %gep_.reloaded12 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded12 = load ptr, ptr %gep_.reloaded12, align 8, !align !1
  %gep_.reloaded13 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded13 = load ptr, ptr %gep_.reloaded13, align 8, !align !1
  %gep_.reloaded14 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded14 = load ptr, ptr %gep_.reloaded14, align 8, !align !2
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !3
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 5
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !3
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i64, align 8
  %p.upperbound = alloca i64, align 8
  %p.stride = alloca i64, align 8
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  %2 = load i64, ptr %loadgep_.reloaded, align 8
  %3 = load i64, ptr %loadgep_.reloaded12, align 8
  %4 = load i64, ptr %loadgep_.reloaded13, align 8
  %5 = load float, ptr %loadgep_.reloaded14, align 4
  br label %omp.region.after_alloca2

omp.region.after_alloca2:                         ; preds = %omp.par.entry
  br label %omp.region.after_alloca

omp.region.after_alloca:                          ; preds = %omp.region.after_alloca2
  br label %omp.par.region

omp.par.region:                                   ; preds = %omp.region.after_alloca
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  br label %omp.wsloop.region

omp.wsloop.region:                                ; preds = %omp.par.region1
  %6 = select i1 false, i64 %2, i64 0
  %7 = select i1 false, i64 0, i64 %2
  %8 = sub nsw i64 %7, %6
  %9 = icmp sle i64 %7, %6
  %10 = sub i64 %8, 1
  %11 = udiv i64 %10, 1
  %12 = add i64 %11, 1
  %13 = icmp ule i64 %8, 1
  %14 = select i1 %13, i64 1, i64 %12
  %omp_loop.tripcount = select i1 %9, i64 0, i64 %14
  br label %omp_loop.preheader

omp_loop.preheader:                               ; preds = %omp.wsloop.region
  store i64 0, ptr %p.lowerbound, align 8
  %15 = sub i64 %omp_loop.tripcount, 1
  store i64 %15, ptr %p.upperbound, align 8
  store i64 1, ptr %p.stride, align 8
  %omp_global_thread_num10 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num10, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i64 1, i64 0)
  %16 = load i64, ptr %p.lowerbound, align 8
  %17 = load i64, ptr %p.upperbound, align 8
  %18 = sub i64 %17, %16
  %19 = add i64 %18, 1
  br label %omp_loop.header

omp_loop.header:                                  ; preds = %omp_loop.inc, %omp_loop.preheader
  %omp_loop.iv = phi i64 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
  br label %omp_loop.cond

omp_loop.cond:                                    ; preds = %omp_loop.header
  %omp_loop.cmp = icmp ult i64 %omp_loop.iv, %19
  br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit

omp_loop.exit:                                    ; preds = %omp_loop.cond
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num10)
  %omp_global_thread_num11 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num11)
  br label %omp_loop.after

omp_loop.after:                                   ; preds = %omp_loop.exit
  br label %omp.region.cont3

omp.region.cont3:                                 ; preds = %omp_loop.after
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %omp.region.cont3
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %omp.region.cont
  br label %.fini

.fini:                                            ; preds = %omp.par.pre_finalize
  br label %omp.par.exit.exitStub

omp_loop.body:                                    ; preds = %omp_loop.cond
  %20 = add i64 %omp_loop.iv, %16
  %21 = mul i64 %20, 1
  %22 = add i64 %21, 0
  br label %omp.loop_nest.region

omp.loop_nest.region:                             ; preds = %omp_loop.body
  %23 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region5

omp.loop_nest.region5:                            ; preds = %omp.loop_nest.region
  br label %omp.loop_nest.region6

omp.loop_nest.region6:                            ; preds = %omp.loop_nest.region7, %omp.loop_nest.region5
  %24 = phi i64 [ %34, %omp.loop_nest.region7 ], [ 0, %omp.loop_nest.region5 ]
  %25 = icmp slt i64 %24, %3
  br i1 %25, label %omp.loop_nest.region7, label %omp.loop_nest.region8

omp.loop_nest.region8:                            ; preds = %omp.loop_nest.region6
  call void @llvm.stackrestore.p0(ptr %23)
  br label %omp.loop_nest.region9

omp.loop_nest.region9:                            ; preds = %omp.loop_nest.region8
  br label %omp.region.cont4

omp.region.cont4:                                 ; preds = %omp.loop_nest.region9
  br label %omp_loop.inc

omp_loop.inc:                                     ; preds = %omp.region.cont4
  %omp_loop.next = add nuw i64 %omp_loop.iv, 1
  br label %omp_loop.header

omp.loop_nest.region7:                            ; preds = %omp.loop_nest.region6
  %26 = mul nuw nsw i64 %22, %4
  %27 = add nuw nsw i64 %26, %24
  %28 = getelementptr inbounds nuw float, ptr %loadgep_, i64 %27
  %29 = load float, ptr %28, align 4
  %30 = fmul float %5, %29
  %31 = mul nuw nsw i64 %22, %3
  %32 = add nuw nsw i64 %31, %24
  %33 = getelementptr inbounds nuw float, ptr %loadgep_2, i64 %32
  store float %30, ptr %33, align 4
  %34 = add i64 %24, 1
  br label %omp.loop_nest.region6

omp.par.exit.exitStub:                            ; preds = %.fini
  ret void
}

define void @_mlir_ciface_k2mm_fp32_polygeist(i64 %0, i64 %1, i64 %2, i64 %3, ptr %4, ptr %5, ptr %6, ptr %7, float %8, float %9) {
  %11 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %4, align 8
  %12 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 0
  %13 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 1
  %14 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 2
  %15 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 3, 0
  %16 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 3, 1
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 4, 0
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 4, 1
  %19 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %5, align 8
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, 0
  %21 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, 1
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, 2
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, 3, 0
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, 3, 1
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, 4, 0
  %26 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, 4, 1
  %27 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %6, align 8
  %28 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, 0
  %29 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, 1
  %30 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, 2
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, 3, 0
  %32 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, 3, 1
  %33 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, 4, 0
  %34 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, 4, 1
  %35 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %7, align 8
  %36 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 0
  %37 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %38 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 2
  %39 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 3, 0
  %40 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 3, 1
  %41 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 4, 0
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 4, 1
  call void @k2mm_fp32_polygeist(i64 %0, i64 %1, i64 %2, i64 %3, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, ptr %28, ptr %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, ptr %36, ptr %37, i64 %38, i64 %39, i64 %40, i64 %41, i64 %42, float %8, float %9)
  ret void
}

define internal void @__npb_alloc_check(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, ptr %6, ptr %7, i64 %8, i64 %9, i64 %10) {
  %12 = getelementptr i8, ptr %1, i64 %2
  %13 = icmp eq ptr %12, null
  %14 = icmp ne i64 %5, 0
  %15 = and i1 %13, %14
  br i1 %15, label %16, label %23

16:                                               ; preds = %11
  %17 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr @stderr, align 8
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 2
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, 1
  %20 = getelementptr { i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, [1 x i8], ptr, i64, ptr, ptr, ptr, ptr, i64, i32, [20 x i8] }, ptr %19, i64 %18
  %21 = getelementptr i8, ptr %7, i64 %8
  %22 = call i32 (ptr, ptr, ...) @fprintf(ptr %20, ptr @str3, i64 %5, ptr %21)
  call void @abort()
  br label %23

23:                                               ; preds = %16, %11
  ret void
}

declare void @abort()

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(ptr) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #1

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_8u(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64) #0

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(ptr, i32) #0

; Function Attrs: convergent nounwind
declare void @__kmpc_barrier(ptr, i32) #2

; Function Attrs: nounwind
declare !callback !4 void @__kmpc_fork_call(ptr, i32, ptr, ...) #0

attributes #0 = { nounwind }
attributes #1 = { nocallback nofree nosync nounwind willreturn }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i64 8}
!2 = !{i64 4}
!3 = !{i64 1}
!4 = !{!5}
!5 = !{i64 2, i64 -1, i64 -1, i1 true}
