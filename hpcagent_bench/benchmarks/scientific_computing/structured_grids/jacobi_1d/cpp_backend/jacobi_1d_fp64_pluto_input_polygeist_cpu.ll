; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 22, ptr @0 }, align 8

define void @jacobi_1d_fp64_polygeist(i64 %0, i64 %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11) {
  %structArg50 = alloca { ptr, ptr, ptr }, align 8
  %structArg = alloca { ptr, ptr, ptr }, align 8
  %.reloaded47 = alloca i64, align 8
  %.reloaded = alloca i64, align 8
  br label %13

13:                                               ; preds = %omp.par.exit15, %12
  %14 = phi i64 [ %19, %omp.par.exit15 ], [ 1, %12 ]
  %15 = icmp slt i64 %14, %1
  br i1 %15, label %16, label %20

16:                                               ; preds = %13
  %17 = add i64 %0, -1
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %17, ptr %.reloaded, align 8
  br label %omp_parallel

omp_parallel:                                     ; preds = %16
  %gep_.reloaded = getelementptr { ptr, ptr, ptr }, ptr %structArg, i32 0, i32 0
  store ptr %.reloaded, ptr %gep_.reloaded, align 8
  %gep_ = getelementptr { ptr, ptr, ptr }, ptr %structArg, i32 0, i32 1
  store ptr %3, ptr %gep_, align 8
  %gep_49 = getelementptr { ptr, ptr, ptr }, ptr %structArg, i32 0, i32 2
  store ptr %8, ptr %gep_49, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @jacobi_1d_fp64_polygeist..omp_par, ptr %structArg)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  %18 = add i64 %0, -1
  %omp_global_thread_num9 = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %18, ptr %.reloaded47, align 8
  br label %omp_parallel53

omp_parallel53:                                   ; preds = %omp.par.exit
  %gep_.reloaded47 = getelementptr { ptr, ptr, ptr }, ptr %structArg50, i32 0, i32 0
  store ptr %.reloaded47, ptr %gep_.reloaded47, align 8
  %gep_51 = getelementptr { ptr, ptr, ptr }, ptr %structArg50, i32 0, i32 1
  store ptr %8, ptr %gep_51, align 8
  %gep_52 = getelementptr { ptr, ptr, ptr }, ptr %structArg50, i32 0, i32 2
  store ptr %3, ptr %gep_52, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @jacobi_1d_fp64_polygeist..omp_par.1, ptr %structArg50)
  br label %omp.par.exit15

omp.par.exit15:                                   ; preds = %omp_parallel53
  %19 = add i64 %14, 1
  br label %13

20:                                               ; preds = %13
  ret void
}

; Function Attrs: nounwind
define internal void @jacobi_1d_fp64_polygeist..omp_par.1(ptr noalias %tid.addr10, ptr noalias %zero.addr11, ptr %0) #0 {
omp.par.entry12:
  %gep_.reloaded47 = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded47 = load ptr, ptr %gep_.reloaded47, align 8, !align !1
  %gep_ = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !2
  %gep_1 = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !2
  %p.lastiter41 = alloca i32, align 4
  %p.lowerbound42 = alloca i64, align 8
  %p.upperbound43 = alloca i64, align 8
  %p.stride44 = alloca i64, align 8
  %tid.addr.local16 = alloca i32, align 4
  %1 = load i32, ptr %tid.addr10, align 4
  store i32 %1, ptr %tid.addr.local16, align 4
  %tid17 = load i32, ptr %tid.addr.local16, align 4
  %2 = load i64, ptr %loadgep_.reloaded47, align 8
  br label %omp.region.after_alloca23

omp.region.after_alloca23:                        ; preds = %omp.par.entry12
  br label %omp.region.after_alloca20

omp.region.after_alloca20:                        ; preds = %omp.region.after_alloca23
  br label %omp.par.region13

omp.par.region13:                                 ; preds = %omp.region.after_alloca20
  br label %omp.par.region22

omp.par.region22:                                 ; preds = %omp.par.region13
  br label %omp.wsloop.region25

omp.wsloop.region25:                              ; preds = %omp.par.region22
  %3 = select i1 false, i64 %2, i64 1
  %4 = select i1 false, i64 1, i64 %2
  %5 = sub nsw i64 %4, %3
  %6 = icmp sle i64 %4, %3
  %7 = sub i64 %5, 1
  %8 = udiv i64 %7, 1
  %9 = add i64 %8, 1
  %10 = icmp ule i64 %5, 1
  %11 = select i1 %10, i64 1, i64 %9
  %omp_loop.tripcount26 = select i1 %6, i64 0, i64 %11
  br label %omp_loop.preheader27

omp_loop.preheader27:                             ; preds = %omp.wsloop.region25
  store i64 0, ptr %p.lowerbound42, align 8
  %12 = sub i64 %omp_loop.tripcount26, 1
  store i64 %12, ptr %p.upperbound43, align 8
  store i64 1, ptr %p.stride44, align 8
  %omp_global_thread_num45 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num45, i32 34, ptr %p.lastiter41, ptr %p.lowerbound42, ptr %p.upperbound43, ptr %p.stride44, i64 1, i64 0)
  %13 = load i64, ptr %p.lowerbound42, align 8
  %14 = load i64, ptr %p.upperbound43, align 8
  %15 = sub i64 %14, %13
  %16 = add i64 %15, 1
  br label %omp_loop.header28

omp_loop.header28:                                ; preds = %omp_loop.inc31, %omp_loop.preheader27
  %omp_loop.iv34 = phi i64 [ 0, %omp_loop.preheader27 ], [ %omp_loop.next36, %omp_loop.inc31 ]
  br label %omp_loop.cond29

omp_loop.cond29:                                  ; preds = %omp_loop.header28
  %omp_loop.cmp35 = icmp ult i64 %omp_loop.iv34, %16
  br i1 %omp_loop.cmp35, label %omp_loop.body30, label %omp_loop.exit32

omp_loop.exit32:                                  ; preds = %omp_loop.cond29
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num45)
  %omp_global_thread_num46 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num46)
  br label %omp_loop.after33

omp_loop.after33:                                 ; preds = %omp_loop.exit32
  br label %omp.region.cont24

omp.region.cont24:                                ; preds = %omp_loop.after33
  br label %omp.region.cont21

omp.region.cont21:                                ; preds = %omp.region.cont24
  br label %omp.par.pre_finalize14

omp.par.pre_finalize14:                           ; preds = %omp.region.cont21
  br label %.fini48

.fini48:                                          ; preds = %omp.par.pre_finalize14
  br label %omp.par.exit15.exitStub

omp_loop.body30:                                  ; preds = %omp_loop.cond29
  %17 = add i64 %omp_loop.iv34, %13
  %18 = mul i64 %17, 1
  %19 = add i64 %18, 1
  br label %omp.loop_nest.region38

omp.loop_nest.region38:                           ; preds = %omp_loop.body30
  %20 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region39

omp.loop_nest.region39:                           ; preds = %omp.loop_nest.region38
  %21 = add i64 %19, -1
  %22 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %21
  %23 = load double, ptr %22, align 8
  %24 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %19
  %25 = load double, ptr %24, align 8
  %26 = fadd double %23, %25
  %27 = add i64 %19, 1
  %28 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %27
  %29 = load double, ptr %28, align 8
  %30 = fadd double %26, %29
  %31 = fmul double %30, 3.333300e-01
  %32 = getelementptr inbounds nuw double, ptr %loadgep_2, i64 %19
  store double %31, ptr %32, align 8
  call void @llvm.stackrestore.p0(ptr %20)
  br label %omp.loop_nest.region40

omp.loop_nest.region40:                           ; preds = %omp.loop_nest.region39
  br label %omp.region.cont37

omp.region.cont37:                                ; preds = %omp.loop_nest.region40
  br label %omp_loop.inc31

omp_loop.inc31:                                   ; preds = %omp.region.cont37
  %omp_loop.next36 = add nuw i64 %omp_loop.iv34, 1
  br label %omp_loop.header28

omp.par.exit15.exitStub:                          ; preds = %.fini48
  ret void
}

; Function Attrs: nounwind
define internal void @jacobi_1d_fp64_polygeist..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #0 {
omp.par.entry:
  %gep_.reloaded = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded = load ptr, ptr %gep_.reloaded, align 8, !align !1
  %gep_ = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !2
  %gep_1 = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !2
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i64, align 8
  %p.upperbound = alloca i64, align 8
  %p.stride = alloca i64, align 8
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  %2 = load i64, ptr %loadgep_.reloaded, align 8
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
  %3 = select i1 false, i64 %2, i64 1
  %4 = select i1 false, i64 1, i64 %2
  %5 = sub nsw i64 %4, %3
  %6 = icmp sle i64 %4, %3
  %7 = sub i64 %5, 1
  %8 = udiv i64 %7, 1
  %9 = add i64 %8, 1
  %10 = icmp ule i64 %5, 1
  %11 = select i1 %10, i64 1, i64 %9
  %omp_loop.tripcount = select i1 %6, i64 0, i64 %11
  br label %omp_loop.preheader

omp_loop.preheader:                               ; preds = %omp.wsloop.region
  store i64 0, ptr %p.lowerbound, align 8
  %12 = sub i64 %omp_loop.tripcount, 1
  store i64 %12, ptr %p.upperbound, align 8
  store i64 1, ptr %p.stride, align 8
  %omp_global_thread_num7 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num7, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i64 1, i64 0)
  %13 = load i64, ptr %p.lowerbound, align 8
  %14 = load i64, ptr %p.upperbound, align 8
  %15 = sub i64 %14, %13
  %16 = add i64 %15, 1
  br label %omp_loop.header

omp_loop.header:                                  ; preds = %omp_loop.inc, %omp_loop.preheader
  %omp_loop.iv = phi i64 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
  br label %omp_loop.cond

omp_loop.cond:                                    ; preds = %omp_loop.header
  %omp_loop.cmp = icmp ult i64 %omp_loop.iv, %16
  br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit

omp_loop.exit:                                    ; preds = %omp_loop.cond
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num7)
  %omp_global_thread_num8 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num8)
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
  %17 = add i64 %omp_loop.iv, %13
  %18 = mul i64 %17, 1
  %19 = add i64 %18, 1
  br label %omp.loop_nest.region

omp.loop_nest.region:                             ; preds = %omp_loop.body
  %20 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region5

omp.loop_nest.region5:                            ; preds = %omp.loop_nest.region
  %21 = add i64 %19, -1
  %22 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %21
  %23 = load double, ptr %22, align 8
  %24 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %19
  %25 = load double, ptr %24, align 8
  %26 = fadd double %23, %25
  %27 = add i64 %19, 1
  %28 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %27
  %29 = load double, ptr %28, align 8
  %30 = fadd double %26, %29
  %31 = fmul double %30, 3.333300e-01
  %32 = getelementptr inbounds nuw double, ptr %loadgep_2, i64 %19
  store double %31, ptr %32, align 8
  call void @llvm.stackrestore.p0(ptr %20)
  br label %omp.loop_nest.region6

omp.loop_nest.region6:                            ; preds = %omp.loop_nest.region5
  br label %omp.region.cont4

omp.region.cont4:                                 ; preds = %omp.loop_nest.region6
  br label %omp_loop.inc

omp_loop.inc:                                     ; preds = %omp.region.cont4
  %omp_loop.next = add nuw i64 %omp_loop.iv, 1
  br label %omp_loop.header

omp.par.exit.exitStub:                            ; preds = %.fini
  ret void
}

define void @_mlir_ciface_jacobi_1d_fp64_polygeist(i64 %0, i64 %1, ptr %2, ptr %3) {
  %5 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %6 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 0
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 1
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 2
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 3, 0
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 4, 0
  %11 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %3, align 8
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 0
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 1
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 2
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 3, 0
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 4, 0
  call void @jacobi_1d_fp64_polygeist(i64 %0, i64 %1, ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16)
  ret void
}

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
declare !callback !3 void @__kmpc_fork_call(ptr, i32, ptr, ...) #0

attributes #0 = { nounwind }
attributes #1 = { nocallback nofree nosync nounwind willreturn }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i64 8}
!2 = !{i64 1}
!3 = !{!4}
!4 = !{i64 2, i64 -1, i64 -1, i1 true}
