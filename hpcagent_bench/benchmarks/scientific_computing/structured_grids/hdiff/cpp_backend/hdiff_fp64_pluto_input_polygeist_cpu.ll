; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@str7 = internal constant [33 x i8] c"out of memory: %zu bytes for %s\0A\00"
@stderr = external global { ptr, ptr, i64, [1 x i64], [1 x i64] }
@str6 = internal constant [10 x i8] c"fly_field\00"
@str5 = internal constant [5 x i8] c"res2\00"
@str4 = internal constant [10 x i8] c"flx_field\00"
@str3 = internal constant [5 x i8] c"res1\00"
@str2 = internal constant [10 x i8] c"lap_field\00"
@str1 = internal constant [6 x i8] c"__cb2\00"
@str0 = internal constant [6 x i8] c"__cb1\00"
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 22, ptr @0 }, align 8

declare void @free(ptr)

declare ptr @malloc(i64)

declare i32 @fprintf(ptr, ptr, ...)

define void @hdiff_fp64_polygeist(i64 %0, i64 %1, i64 %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, ptr %21, ptr %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, i64 %29) {
  %structArg407 = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %structArg403 = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %structArg398 = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %structArg394 = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %structArg390 = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %structArg385 = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %structArg381 = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %structArg = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %.reloaded368 = alloca i64, align 8
  %.reloaded369 = alloca i64, align 8
  %.reloaded370 = alloca i64, align 8
  %.reloaded371 = alloca i64, align 8
  %.reloaded372 = alloca i64, align 8
  %.reloaded373 = alloca i64, align 8
  %.reloaded374 = alloca i64, align 8
  %.reloaded375 = alloca i64, align 8
  %.reloaded376 = alloca i64, align 8
  %.reloaded377 = alloca i64, align 8
  %.reloaded378 = alloca i64, align 8
  %.reloaded318 = alloca i64, align 8
  %.reloaded319 = alloca i64, align 8
  %.reloaded320 = alloca i64, align 8
  %.reloaded321 = alloca i64, align 8
  %.reloaded322 = alloca i64, align 8
  %.reloaded266 = alloca i64, align 8
  %.reloaded267 = alloca i64, align 8
  %.reloaded268 = alloca i64, align 8
  %.reloaded269 = alloca i64, align 8
  %.reloaded270 = alloca i64, align 8
  %.reloaded271 = alloca i64, align 8
  %.reloaded272 = alloca i64, align 8
  %.reloaded216 = alloca i64, align 8
  %.reloaded217 = alloca i64, align 8
  %.reloaded218 = alloca i64, align 8
  %.reloaded219 = alloca i64, align 8
  %.reloaded220 = alloca i64, align 8
  %.reloaded166 = alloca i64, align 8
  %.reloaded167 = alloca i64, align 8
  %.reloaded168 = alloca i64, align 8
  %.reloaded169 = alloca i64, align 8
  %.reloaded170 = alloca i64, align 8
  %.reloaded114 = alloca i64, align 8
  %.reloaded115 = alloca i64, align 8
  %.reloaded116 = alloca i64, align 8
  %.reloaded117 = alloca i64, align 8
  %.reloaded118 = alloca i64, align 8
  %.reloaded119 = alloca i64, align 8
  %.reloaded120 = alloca i64, align 8
  %.reloaded64 = alloca i64, align 8
  %.reloaded65 = alloca i64, align 8
  %.reloaded66 = alloca i64, align 8
  %.reloaded67 = alloca i64, align 8
  %.reloaded68 = alloca i64, align 8
  %.reloaded = alloca i64, align 8
  %.reloaded15 = alloca i64, align 8
  %.reloaded16 = alloca i64, align 8
  %.reloaded17 = alloca i64, align 8
  %.reloaded18 = alloca i64, align 8
  %.reloaded19 = alloca i64, align 8
  %31 = mul i64 %1, %2
  %32 = add i64 %0, 1
  %33 = mul i64 %32, %1
  %34 = mul i64 %33, %2
  %35 = mul i64 %34, 8
  %36 = mul i64 %31, 8
  %37 = udiv i64 %35, %36
  %38 = mul i64 %2, %1
  %39 = mul i64 %38, %37
  %40 = getelementptr double, ptr null, i64 %39
  %41 = ptrtoint ptr %40 to i64
  %42 = call ptr @malloc(i64 %41)
  call void @__npb_alloc_check(ptr %42, ptr %42, i64 0, i64 -9223372036854775808, i64 1, i64 %35, ptr @str0, ptr @str0, i64 0, i64 -9223372036854775808, i64 1)
  %43 = add i64 %1, 1
  %44 = mul i64 %43, %2
  %45 = mul i64 %0, %43
  %46 = mul i64 %45, %2
  %47 = mul i64 %46, 8
  %48 = mul i64 %44, 8
  %49 = udiv i64 %47, %48
  %50 = mul i64 %2, %43
  %51 = mul i64 %50, %49
  %52 = getelementptr double, ptr null, i64 %51
  %53 = ptrtoint ptr %52 to i64
  %54 = call ptr @malloc(i64 %53)
  call void @__npb_alloc_check(ptr %54, ptr %54, i64 0, i64 -9223372036854775808, i64 1, i64 %47, ptr @str1, ptr @str1, i64 0, i64 -9223372036854775808, i64 1)
  %55 = add i64 %1, 2
  %56 = mul i64 %55, %2
  %57 = add i64 %0, 2
  %58 = mul i64 %57, %55
  %59 = mul i64 %58, %2
  %60 = mul i64 %59, 8
  %61 = mul i64 %56, 8
  %62 = udiv i64 %60, %61
  %63 = mul i64 %2, %55
  %64 = mul i64 %63, %62
  %65 = getelementptr double, ptr null, i64 %64
  %66 = ptrtoint ptr %65 to i64
  %67 = call ptr @malloc(i64 %66)
  call void @__npb_alloc_check(ptr %67, ptr %67, i64 0, i64 -9223372036854775808, i64 1, i64 %60, ptr @str2, ptr @str2, i64 0, i64 -9223372036854775808, i64 1)
  %68 = mul i64 %2, %1
  %69 = mul i64 %68, %37
  %70 = getelementptr double, ptr null, i64 %69
  %71 = ptrtoint ptr %70 to i64
  %72 = call ptr @malloc(i64 %71)
  call void @__npb_alloc_check(ptr %72, ptr %72, i64 0, i64 -9223372036854775808, i64 1, i64 %35, ptr @str3, ptr @str3, i64 0, i64 -9223372036854775808, i64 1)
  %73 = mul i64 %2, %1
  %74 = mul i64 %73, %37
  %75 = getelementptr double, ptr null, i64 %74
  %76 = ptrtoint ptr %75 to i64
  %77 = call ptr @malloc(i64 %76)
  call void @__npb_alloc_check(ptr %77, ptr %77, i64 0, i64 -9223372036854775808, i64 1, i64 %35, ptr @str4, ptr @str4, i64 0, i64 -9223372036854775808, i64 1)
  %78 = mul i64 %2, %43
  %79 = mul i64 %78, %49
  %80 = getelementptr double, ptr null, i64 %79
  %81 = ptrtoint ptr %80 to i64
  %82 = call ptr @malloc(i64 %81)
  call void @__npb_alloc_check(ptr %82, ptr %82, i64 0, i64 -9223372036854775808, i64 1, i64 %47, ptr @str5, ptr @str5, i64 0, i64 -9223372036854775808, i64 1)
  %83 = mul i64 %2, %43
  %84 = mul i64 %83, %49
  %85 = getelementptr double, ptr null, i64 %84
  %86 = ptrtoint ptr %85 to i64
  %87 = call ptr @malloc(i64 %86)
  call void @__npb_alloc_check(ptr %87, ptr %87, i64 0, i64 -9223372036854775808, i64 1, i64 %47, ptr @str6, ptr @str6, i64 0, i64 -9223372036854775808, i64 1)
  %88 = add i64 %0, 2
  br label %entry

entry:                                            ; preds = %30
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %88, ptr %.reloaded, align 8
  store i64 %1, ptr %.reloaded15, align 8
  store i64 %2, ptr %.reloaded16, align 8
  store i64 %18, ptr %.reloaded17, align 8
  store i64 %19, ptr %.reloaded18, align 8
  store i64 %63, ptr %.reloaded19, align 8
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_.reloaded = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 0
  store ptr %.reloaded, ptr %gep_.reloaded, align 8
  %gep_.reloaded15 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 1
  store ptr %.reloaded15, ptr %gep_.reloaded15, align 8
  %gep_.reloaded16 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 2
  store ptr %.reloaded16, ptr %gep_.reloaded16, align 8
  %gep_.reloaded17 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 3
  store ptr %.reloaded17, ptr %gep_.reloaded17, align 8
  %gep_.reloaded18 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 4
  store ptr %.reloaded18, ptr %gep_.reloaded18, align 8
  %gep_.reloaded19 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 5
  store ptr %.reloaded19, ptr %gep_.reloaded19, align 8
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 6
  store ptr %13, ptr %gep_, align 8
  %gep_380 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 7
  store ptr %67, ptr %gep_380, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @hdiff_fp64_polygeist..omp_par, ptr %structArg)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  %89 = add i64 %0, 1
  %omp_global_thread_num20 = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %89, ptr %.reloaded64, align 8
  store i64 %1, ptr %.reloaded65, align 8
  store i64 %2, ptr %.reloaded66, align 8
  store i64 %63, ptr %.reloaded67, align 8
  store i64 %68, ptr %.reloaded68, align 8
  br label %omp_parallel384

omp_parallel384:                                  ; preds = %omp.par.exit
  %gep_.reloaded64 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg381, i32 0, i32 0
  store ptr %.reloaded64, ptr %gep_.reloaded64, align 8
  %gep_.reloaded65 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg381, i32 0, i32 1
  store ptr %.reloaded65, ptr %gep_.reloaded65, align 8
  %gep_.reloaded66 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg381, i32 0, i32 2
  store ptr %.reloaded66, ptr %gep_.reloaded66, align 8
  %gep_.reloaded67 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg381, i32 0, i32 3
  store ptr %.reloaded67, ptr %gep_.reloaded67, align 8
  %gep_.reloaded68 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg381, i32 0, i32 4
  store ptr %.reloaded68, ptr %gep_.reloaded68, align 8
  %gep_382 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg381, i32 0, i32 5
  store ptr %67, ptr %gep_382, align 8
  %gep_383 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg381, i32 0, i32 6
  store ptr %72, ptr %gep_383, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @hdiff_fp64_polygeist..omp_par.1, ptr %structArg381)
  br label %omp.par.exit26

omp.par.exit26:                                   ; preds = %omp_parallel384
  %90 = add i64 %0, 1
  %omp_global_thread_num70 = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %90, ptr %.reloaded114, align 8
  store i64 %1, ptr %.reloaded115, align 8
  store i64 %2, ptr %.reloaded116, align 8
  store i64 %68, ptr %.reloaded117, align 8
  store i64 %18, ptr %.reloaded118, align 8
  store i64 %19, ptr %.reloaded119, align 8
  store i64 %38, ptr %.reloaded120, align 8
  br label %omp_parallel389

omp_parallel389:                                  ; preds = %omp.par.exit26
  %gep_.reloaded114 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg385, i32 0, i32 0
  store ptr %.reloaded114, ptr %gep_.reloaded114, align 8
  %gep_.reloaded115 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg385, i32 0, i32 1
  store ptr %.reloaded115, ptr %gep_.reloaded115, align 8
  %gep_.reloaded116 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg385, i32 0, i32 2
  store ptr %.reloaded116, ptr %gep_.reloaded116, align 8
  %gep_.reloaded117 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg385, i32 0, i32 3
  store ptr %.reloaded117, ptr %gep_.reloaded117, align 8
  %gep_.reloaded118 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg385, i32 0, i32 4
  store ptr %.reloaded118, ptr %gep_.reloaded118, align 8
  %gep_.reloaded119 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg385, i32 0, i32 5
  store ptr %.reloaded119, ptr %gep_.reloaded119, align 8
  %gep_.reloaded120 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg385, i32 0, i32 6
  store ptr %.reloaded120, ptr %gep_.reloaded120, align 8
  %gep_386 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg385, i32 0, i32 7
  store ptr %72, ptr %gep_386, align 8
  %gep_387 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg385, i32 0, i32 8
  store ptr %13, ptr %gep_387, align 8
  %gep_388 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg385, i32 0, i32 9
  store ptr %42, ptr %gep_388, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @hdiff_fp64_polygeist..omp_par.2, ptr %structArg385)
  br label %omp.par.exit76

omp.par.exit76:                                   ; preds = %omp_parallel389
  %91 = add i64 %0, 1
  %omp_global_thread_num122 = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %91, ptr %.reloaded166, align 8
  store i64 %1, ptr %.reloaded167, align 8
  store i64 %2, ptr %.reloaded168, align 8
  store i64 %38, ptr %.reloaded169, align 8
  store i64 %73, ptr %.reloaded170, align 8
  br label %omp_parallel393

omp_parallel393:                                  ; preds = %omp.par.exit76
  %gep_.reloaded166 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg390, i32 0, i32 0
  store ptr %.reloaded166, ptr %gep_.reloaded166, align 8
  %gep_.reloaded167 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg390, i32 0, i32 1
  store ptr %.reloaded167, ptr %gep_.reloaded167, align 8
  %gep_.reloaded168 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg390, i32 0, i32 2
  store ptr %.reloaded168, ptr %gep_.reloaded168, align 8
  %gep_.reloaded169 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg390, i32 0, i32 3
  store ptr %.reloaded169, ptr %gep_.reloaded169, align 8
  %gep_.reloaded170 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg390, i32 0, i32 4
  store ptr %.reloaded170, ptr %gep_.reloaded170, align 8
  %gep_391 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg390, i32 0, i32 5
  store ptr %42, ptr %gep_391, align 8
  %gep_392 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg390, i32 0, i32 6
  store ptr %77, ptr %gep_392, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @hdiff_fp64_polygeist..omp_par.3, ptr %structArg390)
  br label %omp.par.exit128

omp.par.exit128:                                  ; preds = %omp_parallel393
  %omp_global_thread_num172 = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %0, ptr %.reloaded216, align 8
  store i64 %1, ptr %.reloaded217, align 8
  store i64 %2, ptr %.reloaded218, align 8
  store i64 %63, ptr %.reloaded219, align 8
  store i64 %78, ptr %.reloaded220, align 8
  br label %omp_parallel397

omp_parallel397:                                  ; preds = %omp.par.exit128
  %gep_.reloaded216 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg394, i32 0, i32 0
  store ptr %.reloaded216, ptr %gep_.reloaded216, align 8
  %gep_.reloaded217 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg394, i32 0, i32 1
  store ptr %.reloaded217, ptr %gep_.reloaded217, align 8
  %gep_.reloaded218 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg394, i32 0, i32 2
  store ptr %.reloaded218, ptr %gep_.reloaded218, align 8
  %gep_.reloaded219 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg394, i32 0, i32 3
  store ptr %.reloaded219, ptr %gep_.reloaded219, align 8
  %gep_.reloaded220 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg394, i32 0, i32 4
  store ptr %.reloaded220, ptr %gep_.reloaded220, align 8
  %gep_395 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg394, i32 0, i32 5
  store ptr %67, ptr %gep_395, align 8
  %gep_396 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg394, i32 0, i32 6
  store ptr %82, ptr %gep_396, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @hdiff_fp64_polygeist..omp_par.4, ptr %structArg394)
  br label %omp.par.exit178

omp.par.exit178:                                  ; preds = %omp_parallel397
  %omp_global_thread_num222 = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %0, ptr %.reloaded266, align 8
  store i64 %1, ptr %.reloaded267, align 8
  store i64 %2, ptr %.reloaded268, align 8
  store i64 %78, ptr %.reloaded269, align 8
  store i64 %18, ptr %.reloaded270, align 8
  store i64 %19, ptr %.reloaded271, align 8
  store i64 %50, ptr %.reloaded272, align 8
  br label %omp_parallel402

omp_parallel402:                                  ; preds = %omp.par.exit178
  %gep_.reloaded266 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg398, i32 0, i32 0
  store ptr %.reloaded266, ptr %gep_.reloaded266, align 8
  %gep_.reloaded267 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg398, i32 0, i32 1
  store ptr %.reloaded267, ptr %gep_.reloaded267, align 8
  %gep_.reloaded268 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg398, i32 0, i32 2
  store ptr %.reloaded268, ptr %gep_.reloaded268, align 8
  %gep_.reloaded269 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg398, i32 0, i32 3
  store ptr %.reloaded269, ptr %gep_.reloaded269, align 8
  %gep_.reloaded270 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg398, i32 0, i32 4
  store ptr %.reloaded270, ptr %gep_.reloaded270, align 8
  %gep_.reloaded271 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg398, i32 0, i32 5
  store ptr %.reloaded271, ptr %gep_.reloaded271, align 8
  %gep_.reloaded272 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg398, i32 0, i32 6
  store ptr %.reloaded272, ptr %gep_.reloaded272, align 8
  %gep_399 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg398, i32 0, i32 7
  store ptr %82, ptr %gep_399, align 8
  %gep_400 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg398, i32 0, i32 8
  store ptr %13, ptr %gep_400, align 8
  %gep_401 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg398, i32 0, i32 9
  store ptr %54, ptr %gep_401, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @hdiff_fp64_polygeist..omp_par.5, ptr %structArg398)
  br label %omp.par.exit228

omp.par.exit228:                                  ; preds = %omp_parallel402
  %omp_global_thread_num274 = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %0, ptr %.reloaded318, align 8
  store i64 %1, ptr %.reloaded319, align 8
  store i64 %2, ptr %.reloaded320, align 8
  store i64 %50, ptr %.reloaded321, align 8
  store i64 %83, ptr %.reloaded322, align 8
  br label %omp_parallel406

omp_parallel406:                                  ; preds = %omp.par.exit228
  %gep_.reloaded318 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg403, i32 0, i32 0
  store ptr %.reloaded318, ptr %gep_.reloaded318, align 8
  %gep_.reloaded319 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg403, i32 0, i32 1
  store ptr %.reloaded319, ptr %gep_.reloaded319, align 8
  %gep_.reloaded320 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg403, i32 0, i32 2
  store ptr %.reloaded320, ptr %gep_.reloaded320, align 8
  %gep_.reloaded321 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg403, i32 0, i32 3
  store ptr %.reloaded321, ptr %gep_.reloaded321, align 8
  %gep_.reloaded322 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg403, i32 0, i32 4
  store ptr %.reloaded322, ptr %gep_.reloaded322, align 8
  %gep_404 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg403, i32 0, i32 5
  store ptr %54, ptr %gep_404, align 8
  %gep_405 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg403, i32 0, i32 6
  store ptr %87, ptr %gep_405, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @hdiff_fp64_polygeist..omp_par.6, ptr %structArg403)
  br label %omp.par.exit280

omp.par.exit280:                                  ; preds = %omp_parallel406
  %omp_global_thread_num324 = call i32 @__kmpc_global_thread_num(ptr @1)
  store i64 %0, ptr %.reloaded368, align 8
  store i64 %1, ptr %.reloaded369, align 8
  store i64 %2, ptr %.reloaded370, align 8
  store i64 %18, ptr %.reloaded371, align 8
  store i64 %19, ptr %.reloaded372, align 8
  store i64 %9, ptr %.reloaded373, align 8
  store i64 %10, ptr %.reloaded374, align 8
  store i64 %73, ptr %.reloaded375, align 8
  store i64 %83, ptr %.reloaded376, align 8
  store i64 %27, ptr %.reloaded377, align 8
  store i64 %28, ptr %.reloaded378, align 8
  br label %omp_parallel413

omp_parallel413:                                  ; preds = %omp.par.exit280
  %gep_.reloaded368 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 0
  store ptr %.reloaded368, ptr %gep_.reloaded368, align 8
  %gep_.reloaded369 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 1
  store ptr %.reloaded369, ptr %gep_.reloaded369, align 8
  %gep_.reloaded370 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 2
  store ptr %.reloaded370, ptr %gep_.reloaded370, align 8
  %gep_.reloaded371 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 3
  store ptr %.reloaded371, ptr %gep_.reloaded371, align 8
  %gep_.reloaded372 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 4
  store ptr %.reloaded372, ptr %gep_.reloaded372, align 8
  %gep_.reloaded373 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 5
  store ptr %.reloaded373, ptr %gep_.reloaded373, align 8
  %gep_.reloaded374 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 6
  store ptr %.reloaded374, ptr %gep_.reloaded374, align 8
  %gep_.reloaded375 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 7
  store ptr %.reloaded375, ptr %gep_.reloaded375, align 8
  %gep_.reloaded376 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 8
  store ptr %.reloaded376, ptr %gep_.reloaded376, align 8
  %gep_.reloaded377 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 9
  store ptr %.reloaded377, ptr %gep_.reloaded377, align 8
  %gep_.reloaded378 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 10
  store ptr %.reloaded378, ptr %gep_.reloaded378, align 8
  %gep_408 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 11
  store ptr %13, ptr %gep_408, align 8
  %gep_409 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 12
  store ptr %4, ptr %gep_409, align 8
  %gep_410 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 13
  store ptr %77, ptr %gep_410, align 8
  %gep_411 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 14
  store ptr %87, ptr %gep_411, align 8
  %gep_412 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %structArg407, i32 0, i32 15
  store ptr %22, ptr %gep_412, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @hdiff_fp64_polygeist..omp_par.7, ptr %structArg407)
  br label %omp.par.exit330

omp.par.exit330:                                  ; preds = %omp_parallel413
  call void @free(ptr %42)
  call void @free(ptr %54)
  call void @free(ptr %67)
  call void @free(ptr %72)
  call void @free(ptr %77)
  call void @free(ptr %82)
  call void @free(ptr %87)
  ret void
}

; Function Attrs: nounwind
define internal void @hdiff_fp64_polygeist..omp_par.7(ptr noalias %tid.addr325, ptr noalias %zero.addr326, ptr %0) #0 {
omp.par.entry327:
  %gep_.reloaded368 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded368 = load ptr, ptr %gep_.reloaded368, align 8, !align !1
  %gep_.reloaded369 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded369 = load ptr, ptr %gep_.reloaded369, align 8, !align !1
  %gep_.reloaded370 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded370 = load ptr, ptr %gep_.reloaded370, align 8, !align !1
  %gep_.reloaded371 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded371 = load ptr, ptr %gep_.reloaded371, align 8, !align !1
  %gep_.reloaded372 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
  %loadgep_.reloaded372 = load ptr, ptr %gep_.reloaded372, align 8, !align !1
  %gep_.reloaded373 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 5
  %loadgep_.reloaded373 = load ptr, ptr %gep_.reloaded373, align 8, !align !1
  %gep_.reloaded374 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 6
  %loadgep_.reloaded374 = load ptr, ptr %gep_.reloaded374, align 8, !align !1
  %gep_.reloaded375 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 7
  %loadgep_.reloaded375 = load ptr, ptr %gep_.reloaded375, align 8, !align !1
  %gep_.reloaded376 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 8
  %loadgep_.reloaded376 = load ptr, ptr %gep_.reloaded376, align 8, !align !1
  %gep_.reloaded377 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 9
  %loadgep_.reloaded377 = load ptr, ptr %gep_.reloaded377, align 8, !align !1
  %gep_.reloaded378 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 10
  %loadgep_.reloaded378 = load ptr, ptr %gep_.reloaded378, align 8, !align !1
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 11
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !2
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 12
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !2
  %gep_3 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 13
  %loadgep_4 = load ptr, ptr %gep_3, align 8, !align !2
  %gep_5 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 14
  %loadgep_6 = load ptr, ptr %gep_5, align 8, !align !2
  %gep_7 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 15
  %loadgep_8 = load ptr, ptr %gep_7, align 8, !align !2
  %p.lastiter362 = alloca i32, align 4
  %p.lowerbound363 = alloca i64, align 8
  %p.upperbound364 = alloca i64, align 8
  %p.stride365 = alloca i64, align 8
  %tid.addr.local331 = alloca i32, align 4
  %1 = load i32, ptr %tid.addr325, align 4
  store i32 %1, ptr %tid.addr.local331, align 4
  %tid332 = load i32, ptr %tid.addr.local331, align 4
  %2 = load i64, ptr %loadgep_.reloaded368, align 8
  %3 = load i64, ptr %loadgep_.reloaded369, align 8
  %4 = load i64, ptr %loadgep_.reloaded370, align 8
  %5 = load i64, ptr %loadgep_.reloaded371, align 8
  %6 = load i64, ptr %loadgep_.reloaded372, align 8
  %7 = load i64, ptr %loadgep_.reloaded373, align 8
  %8 = load i64, ptr %loadgep_.reloaded374, align 8
  %9 = load i64, ptr %loadgep_.reloaded375, align 8
  %10 = load i64, ptr %loadgep_.reloaded376, align 8
  %11 = load i64, ptr %loadgep_.reloaded377, align 8
  %12 = load i64, ptr %loadgep_.reloaded378, align 8
  br label %omp.region.after_alloca338

omp.region.after_alloca338:                       ; preds = %omp.par.entry327
  br label %omp.region.after_alloca335

omp.region.after_alloca335:                       ; preds = %omp.region.after_alloca338
  br label %omp.par.region328

omp.par.region328:                                ; preds = %omp.region.after_alloca335
  br label %omp.par.region337

omp.par.region337:                                ; preds = %omp.par.region328
  br label %omp.wsloop.region340

omp.wsloop.region340:                             ; preds = %omp.par.region337
  %13 = select i1 false, i64 %2, i64 0
  %14 = select i1 false, i64 0, i64 %2
  %15 = sub nsw i64 %14, %13
  %16 = icmp sle i64 %14, %13
  %17 = sub i64 %15, 1
  %18 = udiv i64 %17, 1
  %19 = add i64 %18, 1
  %20 = icmp ule i64 %15, 1
  %21 = select i1 %20, i64 1, i64 %19
  %omp_loop.tripcount341 = select i1 %16, i64 0, i64 %21
  br label %omp_loop.preheader342

omp_loop.preheader342:                            ; preds = %omp.wsloop.region340
  store i64 0, ptr %p.lowerbound363, align 8
  %22 = sub i64 %omp_loop.tripcount341, 1
  store i64 %22, ptr %p.upperbound364, align 8
  store i64 1, ptr %p.stride365, align 8
  %omp_global_thread_num366 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num366, i32 34, ptr %p.lastiter362, ptr %p.lowerbound363, ptr %p.upperbound364, ptr %p.stride365, i64 1, i64 0)
  %23 = load i64, ptr %p.lowerbound363, align 8
  %24 = load i64, ptr %p.upperbound364, align 8
  %25 = sub i64 %24, %23
  %26 = add i64 %25, 1
  br label %omp_loop.header343

omp_loop.header343:                               ; preds = %omp_loop.inc346, %omp_loop.preheader342
  %omp_loop.iv349 = phi i64 [ 0, %omp_loop.preheader342 ], [ %omp_loop.next351, %omp_loop.inc346 ]
  br label %omp_loop.cond344

omp_loop.cond344:                                 ; preds = %omp_loop.header343
  %omp_loop.cmp350 = icmp ult i64 %omp_loop.iv349, %26
  br i1 %omp_loop.cmp350, label %omp_loop.body345, label %omp_loop.exit347

omp_loop.exit347:                                 ; preds = %omp_loop.cond344
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num366)
  %omp_global_thread_num367 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num367)
  br label %omp_loop.after348

omp_loop.after348:                                ; preds = %omp_loop.exit347
  br label %omp.region.cont339

omp.region.cont339:                               ; preds = %omp_loop.after348
  br label %omp.region.cont336

omp.region.cont336:                               ; preds = %omp.region.cont339
  br label %omp.par.pre_finalize329

omp.par.pre_finalize329:                          ; preds = %omp.region.cont336
  br label %.fini379

.fini379:                                         ; preds = %omp.par.pre_finalize329
  br label %omp.par.exit330.exitStub

omp_loop.body345:                                 ; preds = %omp_loop.cond344
  %27 = add i64 %omp_loop.iv349, %23
  %28 = mul i64 %27, 1
  %29 = add i64 %28, 0
  br label %omp.loop_nest.region353

omp.loop_nest.region353:                          ; preds = %omp_loop.body345
  %30 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region354

omp.loop_nest.region354:                          ; preds = %omp.loop_nest.region353
  br label %omp.loop_nest.region355

omp.loop_nest.region355:                          ; preds = %omp.loop_nest.region359, %omp.loop_nest.region354
  %31 = phi i64 [ %35, %omp.loop_nest.region359 ], [ 0, %omp.loop_nest.region354 ]
  %32 = icmp slt i64 %31, %3
  br i1 %32, label %omp.loop_nest.region356, label %omp.loop_nest.region360

omp.loop_nest.region360:                          ; preds = %omp.loop_nest.region355
  call void @llvm.stackrestore.p0(ptr %30)
  br label %omp.loop_nest.region361

omp.loop_nest.region361:                          ; preds = %omp.loop_nest.region360
  br label %omp.region.cont352

omp.region.cont352:                               ; preds = %omp.loop_nest.region361
  br label %omp_loop.inc346

omp_loop.inc346:                                  ; preds = %omp.region.cont352
  %omp_loop.next351 = add nuw i64 %omp_loop.iv349, 1
  br label %omp_loop.header343

omp.loop_nest.region356:                          ; preds = %omp.loop_nest.region355
  br label %omp.loop_nest.region357

omp.loop_nest.region357:                          ; preds = %omp.loop_nest.region358, %omp.loop_nest.region356
  %33 = phi i64 [ %86, %omp.loop_nest.region358 ], [ 0, %omp.loop_nest.region356 ]
  %34 = icmp slt i64 %33, %4
  br i1 %34, label %omp.loop_nest.region358, label %omp.loop_nest.region359

omp.loop_nest.region359:                          ; preds = %omp.loop_nest.region357
  %35 = add i64 %31, 1
  br label %omp.loop_nest.region355

omp.loop_nest.region358:                          ; preds = %omp.loop_nest.region357
  %36 = add i64 %29, 2
  %37 = add i64 %31, 2
  %38 = mul nuw nsw i64 %36, %5
  %39 = mul nuw nsw i64 %37, %6
  %40 = add nuw nsw i64 %38, %39
  %41 = add nuw nsw i64 %40, %33
  %42 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %41
  %43 = load double, ptr %42, align 8
  %44 = mul nuw nsw i64 %29, %7
  %45 = mul nuw nsw i64 %31, %8
  %46 = add nuw nsw i64 %44, %45
  %47 = add nuw nsw i64 %46, %33
  %48 = getelementptr inbounds nuw double, ptr %loadgep_2, i64 %47
  %49 = load double, ptr %48, align 8
  %50 = add i64 %29, 1
  %51 = mul nuw nsw i64 %50, %9
  %52 = mul nuw nsw i64 %31, %4
  %53 = add nuw nsw i64 %51, %52
  %54 = add nuw nsw i64 %53, %33
  %55 = getelementptr inbounds nuw double, ptr %loadgep_4, i64 %54
  %56 = load double, ptr %55, align 8
  %57 = mul nuw nsw i64 %29, %9
  %58 = mul nuw nsw i64 %31, %4
  %59 = add nuw nsw i64 %57, %58
  %60 = add nuw nsw i64 %59, %33
  %61 = getelementptr inbounds nuw double, ptr %loadgep_4, i64 %60
  %62 = load double, ptr %61, align 8
  %63 = fsub double %56, %62
  %64 = add i64 %31, 1
  %65 = mul nuw nsw i64 %29, %10
  %66 = mul nuw nsw i64 %64, %4
  %67 = add nuw nsw i64 %65, %66
  %68 = add nuw nsw i64 %67, %33
  %69 = getelementptr inbounds nuw double, ptr %loadgep_6, i64 %68
  %70 = load double, ptr %69, align 8
  %71 = fadd double %63, %70
  %72 = mul nuw nsw i64 %29, %10
  %73 = mul nuw nsw i64 %31, %4
  %74 = add nuw nsw i64 %72, %73
  %75 = add nuw nsw i64 %74, %33
  %76 = getelementptr inbounds nuw double, ptr %loadgep_6, i64 %75
  %77 = load double, ptr %76, align 8
  %78 = fsub double %71, %77
  %79 = fmul double %49, %78
  %80 = fsub double %43, %79
  %81 = mul nuw nsw i64 %29, %11
  %82 = mul nuw nsw i64 %31, %12
  %83 = add nuw nsw i64 %81, %82
  %84 = add nuw nsw i64 %83, %33
  %85 = getelementptr inbounds nuw double, ptr %loadgep_8, i64 %84
  store double %80, ptr %85, align 8
  %86 = add i64 %33, 1
  br label %omp.loop_nest.region357

omp.par.exit330.exitStub:                         ; preds = %.fini379
  ret void
}

; Function Attrs: nounwind
define internal void @hdiff_fp64_polygeist..omp_par.6(ptr noalias %tid.addr275, ptr noalias %zero.addr276, ptr %0) #0 {
omp.par.entry277:
  %gep_.reloaded318 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded318 = load ptr, ptr %gep_.reloaded318, align 8, !align !1
  %gep_.reloaded319 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded319 = load ptr, ptr %gep_.reloaded319, align 8, !align !1
  %gep_.reloaded320 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded320 = load ptr, ptr %gep_.reloaded320, align 8, !align !1
  %gep_.reloaded321 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded321 = load ptr, ptr %gep_.reloaded321, align 8, !align !1
  %gep_.reloaded322 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
  %loadgep_.reloaded322 = load ptr, ptr %gep_.reloaded322, align 8, !align !1
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 5
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !2
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 6
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !2
  %p.lastiter312 = alloca i32, align 4
  %p.lowerbound313 = alloca i64, align 8
  %p.upperbound314 = alloca i64, align 8
  %p.stride315 = alloca i64, align 8
  %tid.addr.local281 = alloca i32, align 4
  %1 = load i32, ptr %tid.addr275, align 4
  store i32 %1, ptr %tid.addr.local281, align 4
  %tid282 = load i32, ptr %tid.addr.local281, align 4
  %2 = load i64, ptr %loadgep_.reloaded318, align 8
  %3 = load i64, ptr %loadgep_.reloaded319, align 8
  %4 = load i64, ptr %loadgep_.reloaded320, align 8
  %5 = load i64, ptr %loadgep_.reloaded321, align 8
  %6 = load i64, ptr %loadgep_.reloaded322, align 8
  br label %omp.region.after_alloca288

omp.region.after_alloca288:                       ; preds = %omp.par.entry277
  br label %omp.region.after_alloca285

omp.region.after_alloca285:                       ; preds = %omp.region.after_alloca288
  br label %omp.par.region278

omp.par.region278:                                ; preds = %omp.region.after_alloca285
  br label %omp.par.region287

omp.par.region287:                                ; preds = %omp.par.region278
  br label %omp.wsloop.region290

omp.wsloop.region290:                             ; preds = %omp.par.region287
  %7 = select i1 false, i64 %2, i64 0
  %8 = select i1 false, i64 0, i64 %2
  %9 = sub nsw i64 %8, %7
  %10 = icmp sle i64 %8, %7
  %11 = sub i64 %9, 1
  %12 = udiv i64 %11, 1
  %13 = add i64 %12, 1
  %14 = icmp ule i64 %9, 1
  %15 = select i1 %14, i64 1, i64 %13
  %omp_loop.tripcount291 = select i1 %10, i64 0, i64 %15
  br label %omp_loop.preheader292

omp_loop.preheader292:                            ; preds = %omp.wsloop.region290
  store i64 0, ptr %p.lowerbound313, align 8
  %16 = sub i64 %omp_loop.tripcount291, 1
  store i64 %16, ptr %p.upperbound314, align 8
  store i64 1, ptr %p.stride315, align 8
  %omp_global_thread_num316 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num316, i32 34, ptr %p.lastiter312, ptr %p.lowerbound313, ptr %p.upperbound314, ptr %p.stride315, i64 1, i64 0)
  %17 = load i64, ptr %p.lowerbound313, align 8
  %18 = load i64, ptr %p.upperbound314, align 8
  %19 = sub i64 %18, %17
  %20 = add i64 %19, 1
  br label %omp_loop.header293

omp_loop.header293:                               ; preds = %omp_loop.inc296, %omp_loop.preheader292
  %omp_loop.iv299 = phi i64 [ 0, %omp_loop.preheader292 ], [ %omp_loop.next301, %omp_loop.inc296 ]
  br label %omp_loop.cond294

omp_loop.cond294:                                 ; preds = %omp_loop.header293
  %omp_loop.cmp300 = icmp ult i64 %omp_loop.iv299, %20
  br i1 %omp_loop.cmp300, label %omp_loop.body295, label %omp_loop.exit297

omp_loop.exit297:                                 ; preds = %omp_loop.cond294
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num316)
  %omp_global_thread_num317 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num317)
  br label %omp_loop.after298

omp_loop.after298:                                ; preds = %omp_loop.exit297
  br label %omp.region.cont289

omp.region.cont289:                               ; preds = %omp_loop.after298
  br label %omp.region.cont286

omp.region.cont286:                               ; preds = %omp.region.cont289
  br label %omp.par.pre_finalize279

omp.par.pre_finalize279:                          ; preds = %omp.region.cont286
  br label %.fini323

.fini323:                                         ; preds = %omp.par.pre_finalize279
  br label %omp.par.exit280.exitStub

omp_loop.body295:                                 ; preds = %omp_loop.cond294
  %21 = add i64 %omp_loop.iv299, %17
  %22 = mul i64 %21, 1
  %23 = add i64 %22, 0
  br label %omp.loop_nest.region303

omp.loop_nest.region303:                          ; preds = %omp_loop.body295
  %24 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region304

omp.loop_nest.region304:                          ; preds = %omp.loop_nest.region303
  %25 = add i64 %3, 1
  br label %omp.loop_nest.region305

omp.loop_nest.region305:                          ; preds = %omp.loop_nest.region309, %omp.loop_nest.region304
  %26 = phi i64 [ %30, %omp.loop_nest.region309 ], [ 0, %omp.loop_nest.region304 ]
  %27 = icmp slt i64 %26, %25
  br i1 %27, label %omp.loop_nest.region306, label %omp.loop_nest.region310

omp.loop_nest.region310:                          ; preds = %omp.loop_nest.region305
  call void @llvm.stackrestore.p0(ptr %24)
  br label %omp.loop_nest.region311

omp.loop_nest.region311:                          ; preds = %omp.loop_nest.region310
  br label %omp.region.cont302

omp.region.cont302:                               ; preds = %omp.loop_nest.region311
  br label %omp_loop.inc296

omp_loop.inc296:                                  ; preds = %omp.region.cont302
  %omp_loop.next301 = add nuw i64 %omp_loop.iv299, 1
  br label %omp_loop.header293

omp.loop_nest.region306:                          ; preds = %omp.loop_nest.region305
  br label %omp.loop_nest.region307

omp.loop_nest.region307:                          ; preds = %omp.loop_nest.region308, %omp.loop_nest.region306
  %28 = phi i64 [ %42, %omp.loop_nest.region308 ], [ 0, %omp.loop_nest.region306 ]
  %29 = icmp slt i64 %28, %4
  br i1 %29, label %omp.loop_nest.region308, label %omp.loop_nest.region309

omp.loop_nest.region309:                          ; preds = %omp.loop_nest.region307
  %30 = add i64 %26, 1
  br label %omp.loop_nest.region305

omp.loop_nest.region308:                          ; preds = %omp.loop_nest.region307
  %31 = mul nuw nsw i64 %23, %5
  %32 = mul nuw nsw i64 %26, %4
  %33 = add nuw nsw i64 %31, %32
  %34 = add nuw nsw i64 %33, %28
  %35 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %34
  %36 = load double, ptr %35, align 8
  %37 = mul nuw nsw i64 %23, %6
  %38 = mul nuw nsw i64 %26, %4
  %39 = add nuw nsw i64 %37, %38
  %40 = add nuw nsw i64 %39, %28
  %41 = getelementptr inbounds nuw double, ptr %loadgep_2, i64 %40
  store double %36, ptr %41, align 8
  %42 = add i64 %28, 1
  br label %omp.loop_nest.region307

omp.par.exit280.exitStub:                         ; preds = %.fini323
  ret void
}

; Function Attrs: nounwind
define internal void @hdiff_fp64_polygeist..omp_par.5(ptr noalias %tid.addr223, ptr noalias %zero.addr224, ptr %0) #0 {
omp.par.entry225:
  %gep_.reloaded266 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded266 = load ptr, ptr %gep_.reloaded266, align 8, !align !1
  %gep_.reloaded267 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded267 = load ptr, ptr %gep_.reloaded267, align 8, !align !1
  %gep_.reloaded268 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded268 = load ptr, ptr %gep_.reloaded268, align 8, !align !1
  %gep_.reloaded269 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded269 = load ptr, ptr %gep_.reloaded269, align 8, !align !1
  %gep_.reloaded270 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
  %loadgep_.reloaded270 = load ptr, ptr %gep_.reloaded270, align 8, !align !1
  %gep_.reloaded271 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 5
  %loadgep_.reloaded271 = load ptr, ptr %gep_.reloaded271, align 8, !align !1
  %gep_.reloaded272 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 6
  %loadgep_.reloaded272 = load ptr, ptr %gep_.reloaded272, align 8, !align !1
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 7
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !2
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 8
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !2
  %gep_3 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 9
  %loadgep_4 = load ptr, ptr %gep_3, align 8, !align !2
  %p.lastiter260 = alloca i32, align 4
  %p.lowerbound261 = alloca i64, align 8
  %p.upperbound262 = alloca i64, align 8
  %p.stride263 = alloca i64, align 8
  %tid.addr.local229 = alloca i32, align 4
  %1 = load i32, ptr %tid.addr223, align 4
  store i32 %1, ptr %tid.addr.local229, align 4
  %tid230 = load i32, ptr %tid.addr.local229, align 4
  %2 = load i64, ptr %loadgep_.reloaded266, align 8
  %3 = load i64, ptr %loadgep_.reloaded267, align 8
  %4 = load i64, ptr %loadgep_.reloaded268, align 8
  %5 = load i64, ptr %loadgep_.reloaded269, align 8
  %6 = load i64, ptr %loadgep_.reloaded270, align 8
  %7 = load i64, ptr %loadgep_.reloaded271, align 8
  %8 = load i64, ptr %loadgep_.reloaded272, align 8
  br label %omp.region.after_alloca236

omp.region.after_alloca236:                       ; preds = %omp.par.entry225
  br label %omp.region.after_alloca233

omp.region.after_alloca233:                       ; preds = %omp.region.after_alloca236
  br label %omp.par.region226

omp.par.region226:                                ; preds = %omp.region.after_alloca233
  br label %omp.par.region235

omp.par.region235:                                ; preds = %omp.par.region226
  br label %omp.wsloop.region238

omp.wsloop.region238:                             ; preds = %omp.par.region235
  %9 = select i1 false, i64 %2, i64 0
  %10 = select i1 false, i64 0, i64 %2
  %11 = sub nsw i64 %10, %9
  %12 = icmp sle i64 %10, %9
  %13 = sub i64 %11, 1
  %14 = udiv i64 %13, 1
  %15 = add i64 %14, 1
  %16 = icmp ule i64 %11, 1
  %17 = select i1 %16, i64 1, i64 %15
  %omp_loop.tripcount239 = select i1 %12, i64 0, i64 %17
  br label %omp_loop.preheader240

omp_loop.preheader240:                            ; preds = %omp.wsloop.region238
  store i64 0, ptr %p.lowerbound261, align 8
  %18 = sub i64 %omp_loop.tripcount239, 1
  store i64 %18, ptr %p.upperbound262, align 8
  store i64 1, ptr %p.stride263, align 8
  %omp_global_thread_num264 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num264, i32 34, ptr %p.lastiter260, ptr %p.lowerbound261, ptr %p.upperbound262, ptr %p.stride263, i64 1, i64 0)
  %19 = load i64, ptr %p.lowerbound261, align 8
  %20 = load i64, ptr %p.upperbound262, align 8
  %21 = sub i64 %20, %19
  %22 = add i64 %21, 1
  br label %omp_loop.header241

omp_loop.header241:                               ; preds = %omp_loop.inc244, %omp_loop.preheader240
  %omp_loop.iv247 = phi i64 [ 0, %omp_loop.preheader240 ], [ %omp_loop.next249, %omp_loop.inc244 ]
  br label %omp_loop.cond242

omp_loop.cond242:                                 ; preds = %omp_loop.header241
  %omp_loop.cmp248 = icmp ult i64 %omp_loop.iv247, %22
  br i1 %omp_loop.cmp248, label %omp_loop.body243, label %omp_loop.exit245

omp_loop.exit245:                                 ; preds = %omp_loop.cond242
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num264)
  %omp_global_thread_num265 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num265)
  br label %omp_loop.after246

omp_loop.after246:                                ; preds = %omp_loop.exit245
  br label %omp.region.cont237

omp.region.cont237:                               ; preds = %omp_loop.after246
  br label %omp.region.cont234

omp.region.cont234:                               ; preds = %omp.region.cont237
  br label %omp.par.pre_finalize227

omp.par.pre_finalize227:                          ; preds = %omp.region.cont234
  br label %.fini273

.fini273:                                         ; preds = %omp.par.pre_finalize227
  br label %omp.par.exit228.exitStub

omp_loop.body243:                                 ; preds = %omp_loop.cond242
  %23 = add i64 %omp_loop.iv247, %19
  %24 = mul i64 %23, 1
  %25 = add i64 %24, 0
  br label %omp.loop_nest.region251

omp.loop_nest.region251:                          ; preds = %omp_loop.body243
  %26 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region252

omp.loop_nest.region252:                          ; preds = %omp.loop_nest.region251
  %27 = add i64 %3, 1
  br label %omp.loop_nest.region253

omp.loop_nest.region253:                          ; preds = %omp.loop_nest.region257, %omp.loop_nest.region252
  %28 = phi i64 [ %32, %omp.loop_nest.region257 ], [ 0, %omp.loop_nest.region252 ]
  %29 = icmp slt i64 %28, %27
  br i1 %29, label %omp.loop_nest.region254, label %omp.loop_nest.region258

omp.loop_nest.region258:                          ; preds = %omp.loop_nest.region253
  call void @llvm.stackrestore.p0(ptr %26)
  br label %omp.loop_nest.region259

omp.loop_nest.region259:                          ; preds = %omp.loop_nest.region258
  br label %omp.region.cont250

omp.region.cont250:                               ; preds = %omp.loop_nest.region259
  br label %omp_loop.inc244

omp_loop.inc244:                                  ; preds = %omp.region.cont250
  %omp_loop.next249 = add nuw i64 %omp_loop.iv247, 1
  br label %omp_loop.header241

omp.loop_nest.region254:                          ; preds = %omp.loop_nest.region253
  br label %omp.loop_nest.region255

omp.loop_nest.region255:                          ; preds = %omp.loop_nest.region256, %omp.loop_nest.region254
  %30 = phi i64 [ %64, %omp.loop_nest.region256 ], [ 0, %omp.loop_nest.region254 ]
  %31 = icmp slt i64 %30, %4
  br i1 %31, label %omp.loop_nest.region256, label %omp.loop_nest.region257

omp.loop_nest.region257:                          ; preds = %omp.loop_nest.region255
  %32 = add i64 %28, 1
  br label %omp.loop_nest.region253

omp.loop_nest.region256:                          ; preds = %omp.loop_nest.region255
  %33 = mul nuw nsw i64 %25, %5
  %34 = mul nuw nsw i64 %28, %4
  %35 = add nuw nsw i64 %33, %34
  %36 = add nuw nsw i64 %35, %30
  %37 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %36
  %38 = load double, ptr %37, align 8
  %39 = add i64 %25, 2
  %40 = add i64 %28, 2
  %41 = mul nuw nsw i64 %39, %6
  %42 = mul nuw nsw i64 %40, %7
  %43 = add nuw nsw i64 %41, %42
  %44 = add nuw nsw i64 %43, %30
  %45 = getelementptr inbounds nuw double, ptr %loadgep_2, i64 %44
  %46 = load double, ptr %45, align 8
  %47 = add i64 %25, 2
  %48 = add i64 %28, 1
  %49 = mul nuw nsw i64 %47, %6
  %50 = mul nuw nsw i64 %48, %7
  %51 = add nuw nsw i64 %49, %50
  %52 = add nuw nsw i64 %51, %30
  %53 = getelementptr inbounds nuw double, ptr %loadgep_2, i64 %52
  %54 = load double, ptr %53, align 8
  %55 = fsub double %46, %54
  %56 = fmul double %38, %55
  %57 = fcmp ogt double %56, 0.000000e+00
  %58 = select i1 %57, double 0.000000e+00, double %38
  %59 = mul nuw nsw i64 %25, %8
  %60 = mul nuw nsw i64 %28, %4
  %61 = add nuw nsw i64 %59, %60
  %62 = add nuw nsw i64 %61, %30
  %63 = getelementptr inbounds nuw double, ptr %loadgep_4, i64 %62
  store double %58, ptr %63, align 8
  %64 = add i64 %30, 1
  br label %omp.loop_nest.region255

omp.par.exit228.exitStub:                         ; preds = %.fini273
  ret void
}

; Function Attrs: nounwind
define internal void @hdiff_fp64_polygeist..omp_par.4(ptr noalias %tid.addr173, ptr noalias %zero.addr174, ptr %0) #0 {
omp.par.entry175:
  %gep_.reloaded216 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded216 = load ptr, ptr %gep_.reloaded216, align 8, !align !1
  %gep_.reloaded217 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded217 = load ptr, ptr %gep_.reloaded217, align 8, !align !1
  %gep_.reloaded218 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded218 = load ptr, ptr %gep_.reloaded218, align 8, !align !1
  %gep_.reloaded219 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded219 = load ptr, ptr %gep_.reloaded219, align 8, !align !1
  %gep_.reloaded220 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
  %loadgep_.reloaded220 = load ptr, ptr %gep_.reloaded220, align 8, !align !1
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 5
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !2
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 6
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !2
  %p.lastiter210 = alloca i32, align 4
  %p.lowerbound211 = alloca i64, align 8
  %p.upperbound212 = alloca i64, align 8
  %p.stride213 = alloca i64, align 8
  %tid.addr.local179 = alloca i32, align 4
  %1 = load i32, ptr %tid.addr173, align 4
  store i32 %1, ptr %tid.addr.local179, align 4
  %tid180 = load i32, ptr %tid.addr.local179, align 4
  %2 = load i64, ptr %loadgep_.reloaded216, align 8
  %3 = load i64, ptr %loadgep_.reloaded217, align 8
  %4 = load i64, ptr %loadgep_.reloaded218, align 8
  %5 = load i64, ptr %loadgep_.reloaded219, align 8
  %6 = load i64, ptr %loadgep_.reloaded220, align 8
  br label %omp.region.after_alloca186

omp.region.after_alloca186:                       ; preds = %omp.par.entry175
  br label %omp.region.after_alloca183

omp.region.after_alloca183:                       ; preds = %omp.region.after_alloca186
  br label %omp.par.region176

omp.par.region176:                                ; preds = %omp.region.after_alloca183
  br label %omp.par.region185

omp.par.region185:                                ; preds = %omp.par.region176
  br label %omp.wsloop.region188

omp.wsloop.region188:                             ; preds = %omp.par.region185
  %7 = select i1 false, i64 %2, i64 0
  %8 = select i1 false, i64 0, i64 %2
  %9 = sub nsw i64 %8, %7
  %10 = icmp sle i64 %8, %7
  %11 = sub i64 %9, 1
  %12 = udiv i64 %11, 1
  %13 = add i64 %12, 1
  %14 = icmp ule i64 %9, 1
  %15 = select i1 %14, i64 1, i64 %13
  %omp_loop.tripcount189 = select i1 %10, i64 0, i64 %15
  br label %omp_loop.preheader190

omp_loop.preheader190:                            ; preds = %omp.wsloop.region188
  store i64 0, ptr %p.lowerbound211, align 8
  %16 = sub i64 %omp_loop.tripcount189, 1
  store i64 %16, ptr %p.upperbound212, align 8
  store i64 1, ptr %p.stride213, align 8
  %omp_global_thread_num214 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num214, i32 34, ptr %p.lastiter210, ptr %p.lowerbound211, ptr %p.upperbound212, ptr %p.stride213, i64 1, i64 0)
  %17 = load i64, ptr %p.lowerbound211, align 8
  %18 = load i64, ptr %p.upperbound212, align 8
  %19 = sub i64 %18, %17
  %20 = add i64 %19, 1
  br label %omp_loop.header191

omp_loop.header191:                               ; preds = %omp_loop.inc194, %omp_loop.preheader190
  %omp_loop.iv197 = phi i64 [ 0, %omp_loop.preheader190 ], [ %omp_loop.next199, %omp_loop.inc194 ]
  br label %omp_loop.cond192

omp_loop.cond192:                                 ; preds = %omp_loop.header191
  %omp_loop.cmp198 = icmp ult i64 %omp_loop.iv197, %20
  br i1 %omp_loop.cmp198, label %omp_loop.body193, label %omp_loop.exit195

omp_loop.exit195:                                 ; preds = %omp_loop.cond192
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num214)
  %omp_global_thread_num215 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num215)
  br label %omp_loop.after196

omp_loop.after196:                                ; preds = %omp_loop.exit195
  br label %omp.region.cont187

omp.region.cont187:                               ; preds = %omp_loop.after196
  br label %omp.region.cont184

omp.region.cont184:                               ; preds = %omp.region.cont187
  br label %omp.par.pre_finalize177

omp.par.pre_finalize177:                          ; preds = %omp.region.cont184
  br label %.fini221

.fini221:                                         ; preds = %omp.par.pre_finalize177
  br label %omp.par.exit178.exitStub

omp_loop.body193:                                 ; preds = %omp_loop.cond192
  %21 = add i64 %omp_loop.iv197, %17
  %22 = mul i64 %21, 1
  %23 = add i64 %22, 0
  br label %omp.loop_nest.region201

omp.loop_nest.region201:                          ; preds = %omp_loop.body193
  %24 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region202

omp.loop_nest.region202:                          ; preds = %omp.loop_nest.region201
  %25 = add i64 %3, 1
  br label %omp.loop_nest.region203

omp.loop_nest.region203:                          ; preds = %omp.loop_nest.region207, %omp.loop_nest.region202
  %26 = phi i64 [ %30, %omp.loop_nest.region207 ], [ 0, %omp.loop_nest.region202 ]
  %27 = icmp slt i64 %26, %25
  br i1 %27, label %omp.loop_nest.region204, label %omp.loop_nest.region208

omp.loop_nest.region208:                          ; preds = %omp.loop_nest.region203
  call void @llvm.stackrestore.p0(ptr %24)
  br label %omp.loop_nest.region209

omp.loop_nest.region209:                          ; preds = %omp.loop_nest.region208
  br label %omp.region.cont200

omp.region.cont200:                               ; preds = %omp.loop_nest.region209
  br label %omp_loop.inc194

omp_loop.inc194:                                  ; preds = %omp.region.cont200
  %omp_loop.next199 = add nuw i64 %omp_loop.iv197, 1
  br label %omp_loop.header191

omp.loop_nest.region204:                          ; preds = %omp.loop_nest.region203
  br label %omp.loop_nest.region205

omp.loop_nest.region205:                          ; preds = %omp.loop_nest.region206, %omp.loop_nest.region204
  %28 = phi i64 [ %52, %omp.loop_nest.region206 ], [ 0, %omp.loop_nest.region204 ]
  %29 = icmp slt i64 %28, %4
  br i1 %29, label %omp.loop_nest.region206, label %omp.loop_nest.region207

omp.loop_nest.region207:                          ; preds = %omp.loop_nest.region205
  %30 = add i64 %26, 1
  br label %omp.loop_nest.region203

omp.loop_nest.region206:                          ; preds = %omp.loop_nest.region205
  %31 = add i64 %23, 1
  %32 = add i64 %26, 1
  %33 = mul nuw nsw i64 %31, %5
  %34 = mul nuw nsw i64 %32, %4
  %35 = add nuw nsw i64 %33, %34
  %36 = add nuw nsw i64 %35, %28
  %37 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %36
  %38 = load double, ptr %37, align 8
  %39 = add i64 %23, 1
  %40 = mul nuw nsw i64 %39, %5
  %41 = mul nuw nsw i64 %26, %4
  %42 = add nuw nsw i64 %40, %41
  %43 = add nuw nsw i64 %42, %28
  %44 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %43
  %45 = load double, ptr %44, align 8
  %46 = fsub double %38, %45
  %47 = mul nuw nsw i64 %23, %6
  %48 = mul nuw nsw i64 %26, %4
  %49 = add nuw nsw i64 %47, %48
  %50 = add nuw nsw i64 %49, %28
  %51 = getelementptr inbounds nuw double, ptr %loadgep_2, i64 %50
  store double %46, ptr %51, align 8
  %52 = add i64 %28, 1
  br label %omp.loop_nest.region205

omp.par.exit178.exitStub:                         ; preds = %.fini221
  ret void
}

; Function Attrs: nounwind
define internal void @hdiff_fp64_polygeist..omp_par.3(ptr noalias %tid.addr123, ptr noalias %zero.addr124, ptr %0) #0 {
omp.par.entry125:
  %gep_.reloaded166 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded166 = load ptr, ptr %gep_.reloaded166, align 8, !align !1
  %gep_.reloaded167 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded167 = load ptr, ptr %gep_.reloaded167, align 8, !align !1
  %gep_.reloaded168 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded168 = load ptr, ptr %gep_.reloaded168, align 8, !align !1
  %gep_.reloaded169 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded169 = load ptr, ptr %gep_.reloaded169, align 8, !align !1
  %gep_.reloaded170 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
  %loadgep_.reloaded170 = load ptr, ptr %gep_.reloaded170, align 8, !align !1
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 5
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !2
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 6
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !2
  %p.lastiter160 = alloca i32, align 4
  %p.lowerbound161 = alloca i64, align 8
  %p.upperbound162 = alloca i64, align 8
  %p.stride163 = alloca i64, align 8
  %tid.addr.local129 = alloca i32, align 4
  %1 = load i32, ptr %tid.addr123, align 4
  store i32 %1, ptr %tid.addr.local129, align 4
  %tid130 = load i32, ptr %tid.addr.local129, align 4
  %2 = load i64, ptr %loadgep_.reloaded166, align 8
  %3 = load i64, ptr %loadgep_.reloaded167, align 8
  %4 = load i64, ptr %loadgep_.reloaded168, align 8
  %5 = load i64, ptr %loadgep_.reloaded169, align 8
  %6 = load i64, ptr %loadgep_.reloaded170, align 8
  br label %omp.region.after_alloca136

omp.region.after_alloca136:                       ; preds = %omp.par.entry125
  br label %omp.region.after_alloca133

omp.region.after_alloca133:                       ; preds = %omp.region.after_alloca136
  br label %omp.par.region126

omp.par.region126:                                ; preds = %omp.region.after_alloca133
  br label %omp.par.region135

omp.par.region135:                                ; preds = %omp.par.region126
  br label %omp.wsloop.region138

omp.wsloop.region138:                             ; preds = %omp.par.region135
  %7 = select i1 false, i64 %2, i64 0
  %8 = select i1 false, i64 0, i64 %2
  %9 = sub nsw i64 %8, %7
  %10 = icmp sle i64 %8, %7
  %11 = sub i64 %9, 1
  %12 = udiv i64 %11, 1
  %13 = add i64 %12, 1
  %14 = icmp ule i64 %9, 1
  %15 = select i1 %14, i64 1, i64 %13
  %omp_loop.tripcount139 = select i1 %10, i64 0, i64 %15
  br label %omp_loop.preheader140

omp_loop.preheader140:                            ; preds = %omp.wsloop.region138
  store i64 0, ptr %p.lowerbound161, align 8
  %16 = sub i64 %omp_loop.tripcount139, 1
  store i64 %16, ptr %p.upperbound162, align 8
  store i64 1, ptr %p.stride163, align 8
  %omp_global_thread_num164 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num164, i32 34, ptr %p.lastiter160, ptr %p.lowerbound161, ptr %p.upperbound162, ptr %p.stride163, i64 1, i64 0)
  %17 = load i64, ptr %p.lowerbound161, align 8
  %18 = load i64, ptr %p.upperbound162, align 8
  %19 = sub i64 %18, %17
  %20 = add i64 %19, 1
  br label %omp_loop.header141

omp_loop.header141:                               ; preds = %omp_loop.inc144, %omp_loop.preheader140
  %omp_loop.iv147 = phi i64 [ 0, %omp_loop.preheader140 ], [ %omp_loop.next149, %omp_loop.inc144 ]
  br label %omp_loop.cond142

omp_loop.cond142:                                 ; preds = %omp_loop.header141
  %omp_loop.cmp148 = icmp ult i64 %omp_loop.iv147, %20
  br i1 %omp_loop.cmp148, label %omp_loop.body143, label %omp_loop.exit145

omp_loop.exit145:                                 ; preds = %omp_loop.cond142
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num164)
  %omp_global_thread_num165 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num165)
  br label %omp_loop.after146

omp_loop.after146:                                ; preds = %omp_loop.exit145
  br label %omp.region.cont137

omp.region.cont137:                               ; preds = %omp_loop.after146
  br label %omp.region.cont134

omp.region.cont134:                               ; preds = %omp.region.cont137
  br label %omp.par.pre_finalize127

omp.par.pre_finalize127:                          ; preds = %omp.region.cont134
  br label %.fini171

.fini171:                                         ; preds = %omp.par.pre_finalize127
  br label %omp.par.exit128.exitStub

omp_loop.body143:                                 ; preds = %omp_loop.cond142
  %21 = add i64 %omp_loop.iv147, %17
  %22 = mul i64 %21, 1
  %23 = add i64 %22, 0
  br label %omp.loop_nest.region151

omp.loop_nest.region151:                          ; preds = %omp_loop.body143
  %24 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region152

omp.loop_nest.region152:                          ; preds = %omp.loop_nest.region151
  br label %omp.loop_nest.region153

omp.loop_nest.region153:                          ; preds = %omp.loop_nest.region157, %omp.loop_nest.region152
  %25 = phi i64 [ %29, %omp.loop_nest.region157 ], [ 0, %omp.loop_nest.region152 ]
  %26 = icmp slt i64 %25, %3
  br i1 %26, label %omp.loop_nest.region154, label %omp.loop_nest.region158

omp.loop_nest.region158:                          ; preds = %omp.loop_nest.region153
  call void @llvm.stackrestore.p0(ptr %24)
  br label %omp.loop_nest.region159

omp.loop_nest.region159:                          ; preds = %omp.loop_nest.region158
  br label %omp.region.cont150

omp.region.cont150:                               ; preds = %omp.loop_nest.region159
  br label %omp_loop.inc144

omp_loop.inc144:                                  ; preds = %omp.region.cont150
  %omp_loop.next149 = add nuw i64 %omp_loop.iv147, 1
  br label %omp_loop.header141

omp.loop_nest.region154:                          ; preds = %omp.loop_nest.region153
  br label %omp.loop_nest.region155

omp.loop_nest.region155:                          ; preds = %omp.loop_nest.region156, %omp.loop_nest.region154
  %27 = phi i64 [ %41, %omp.loop_nest.region156 ], [ 0, %omp.loop_nest.region154 ]
  %28 = icmp slt i64 %27, %4
  br i1 %28, label %omp.loop_nest.region156, label %omp.loop_nest.region157

omp.loop_nest.region157:                          ; preds = %omp.loop_nest.region155
  %29 = add i64 %25, 1
  br label %omp.loop_nest.region153

omp.loop_nest.region156:                          ; preds = %omp.loop_nest.region155
  %30 = mul nuw nsw i64 %23, %5
  %31 = mul nuw nsw i64 %25, %4
  %32 = add nuw nsw i64 %30, %31
  %33 = add nuw nsw i64 %32, %27
  %34 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %33
  %35 = load double, ptr %34, align 8
  %36 = mul nuw nsw i64 %23, %6
  %37 = mul nuw nsw i64 %25, %4
  %38 = add nuw nsw i64 %36, %37
  %39 = add nuw nsw i64 %38, %27
  %40 = getelementptr inbounds nuw double, ptr %loadgep_2, i64 %39
  store double %35, ptr %40, align 8
  %41 = add i64 %27, 1
  br label %omp.loop_nest.region155

omp.par.exit128.exitStub:                         ; preds = %.fini171
  ret void
}

; Function Attrs: nounwind
define internal void @hdiff_fp64_polygeist..omp_par.2(ptr noalias %tid.addr71, ptr noalias %zero.addr72, ptr %0) #0 {
omp.par.entry73:
  %gep_.reloaded114 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded114 = load ptr, ptr %gep_.reloaded114, align 8, !align !1
  %gep_.reloaded115 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded115 = load ptr, ptr %gep_.reloaded115, align 8, !align !1
  %gep_.reloaded116 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded116 = load ptr, ptr %gep_.reloaded116, align 8, !align !1
  %gep_.reloaded117 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded117 = load ptr, ptr %gep_.reloaded117, align 8, !align !1
  %gep_.reloaded118 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
  %loadgep_.reloaded118 = load ptr, ptr %gep_.reloaded118, align 8, !align !1
  %gep_.reloaded119 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 5
  %loadgep_.reloaded119 = load ptr, ptr %gep_.reloaded119, align 8, !align !1
  %gep_.reloaded120 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 6
  %loadgep_.reloaded120 = load ptr, ptr %gep_.reloaded120, align 8, !align !1
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 7
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !2
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 8
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !2
  %gep_3 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 9
  %loadgep_4 = load ptr, ptr %gep_3, align 8, !align !2
  %p.lastiter108 = alloca i32, align 4
  %p.lowerbound109 = alloca i64, align 8
  %p.upperbound110 = alloca i64, align 8
  %p.stride111 = alloca i64, align 8
  %tid.addr.local77 = alloca i32, align 4
  %1 = load i32, ptr %tid.addr71, align 4
  store i32 %1, ptr %tid.addr.local77, align 4
  %tid78 = load i32, ptr %tid.addr.local77, align 4
  %2 = load i64, ptr %loadgep_.reloaded114, align 8
  %3 = load i64, ptr %loadgep_.reloaded115, align 8
  %4 = load i64, ptr %loadgep_.reloaded116, align 8
  %5 = load i64, ptr %loadgep_.reloaded117, align 8
  %6 = load i64, ptr %loadgep_.reloaded118, align 8
  %7 = load i64, ptr %loadgep_.reloaded119, align 8
  %8 = load i64, ptr %loadgep_.reloaded120, align 8
  br label %omp.region.after_alloca84

omp.region.after_alloca84:                        ; preds = %omp.par.entry73
  br label %omp.region.after_alloca81

omp.region.after_alloca81:                        ; preds = %omp.region.after_alloca84
  br label %omp.par.region74

omp.par.region74:                                 ; preds = %omp.region.after_alloca81
  br label %omp.par.region83

omp.par.region83:                                 ; preds = %omp.par.region74
  br label %omp.wsloop.region86

omp.wsloop.region86:                              ; preds = %omp.par.region83
  %9 = select i1 false, i64 %2, i64 0
  %10 = select i1 false, i64 0, i64 %2
  %11 = sub nsw i64 %10, %9
  %12 = icmp sle i64 %10, %9
  %13 = sub i64 %11, 1
  %14 = udiv i64 %13, 1
  %15 = add i64 %14, 1
  %16 = icmp ule i64 %11, 1
  %17 = select i1 %16, i64 1, i64 %15
  %omp_loop.tripcount87 = select i1 %12, i64 0, i64 %17
  br label %omp_loop.preheader88

omp_loop.preheader88:                             ; preds = %omp.wsloop.region86
  store i64 0, ptr %p.lowerbound109, align 8
  %18 = sub i64 %omp_loop.tripcount87, 1
  store i64 %18, ptr %p.upperbound110, align 8
  store i64 1, ptr %p.stride111, align 8
  %omp_global_thread_num112 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num112, i32 34, ptr %p.lastiter108, ptr %p.lowerbound109, ptr %p.upperbound110, ptr %p.stride111, i64 1, i64 0)
  %19 = load i64, ptr %p.lowerbound109, align 8
  %20 = load i64, ptr %p.upperbound110, align 8
  %21 = sub i64 %20, %19
  %22 = add i64 %21, 1
  br label %omp_loop.header89

omp_loop.header89:                                ; preds = %omp_loop.inc92, %omp_loop.preheader88
  %omp_loop.iv95 = phi i64 [ 0, %omp_loop.preheader88 ], [ %omp_loop.next97, %omp_loop.inc92 ]
  br label %omp_loop.cond90

omp_loop.cond90:                                  ; preds = %omp_loop.header89
  %omp_loop.cmp96 = icmp ult i64 %omp_loop.iv95, %22
  br i1 %omp_loop.cmp96, label %omp_loop.body91, label %omp_loop.exit93

omp_loop.exit93:                                  ; preds = %omp_loop.cond90
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num112)
  %omp_global_thread_num113 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num113)
  br label %omp_loop.after94

omp_loop.after94:                                 ; preds = %omp_loop.exit93
  br label %omp.region.cont85

omp.region.cont85:                                ; preds = %omp_loop.after94
  br label %omp.region.cont82

omp.region.cont82:                                ; preds = %omp.region.cont85
  br label %omp.par.pre_finalize75

omp.par.pre_finalize75:                           ; preds = %omp.region.cont82
  br label %.fini121

.fini121:                                         ; preds = %omp.par.pre_finalize75
  br label %omp.par.exit76.exitStub

omp_loop.body91:                                  ; preds = %omp_loop.cond90
  %23 = add i64 %omp_loop.iv95, %19
  %24 = mul i64 %23, 1
  %25 = add i64 %24, 0
  br label %omp.loop_nest.region99

omp.loop_nest.region99:                           ; preds = %omp_loop.body91
  %26 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region100

omp.loop_nest.region100:                          ; preds = %omp.loop_nest.region99
  br label %omp.loop_nest.region101

omp.loop_nest.region101:                          ; preds = %omp.loop_nest.region105, %omp.loop_nest.region100
  %27 = phi i64 [ %31, %omp.loop_nest.region105 ], [ 0, %omp.loop_nest.region100 ]
  %28 = icmp slt i64 %27, %3
  br i1 %28, label %omp.loop_nest.region102, label %omp.loop_nest.region106

omp.loop_nest.region106:                          ; preds = %omp.loop_nest.region101
  call void @llvm.stackrestore.p0(ptr %26)
  br label %omp.loop_nest.region107

omp.loop_nest.region107:                          ; preds = %omp.loop_nest.region106
  br label %omp.region.cont98

omp.region.cont98:                                ; preds = %omp.loop_nest.region107
  br label %omp_loop.inc92

omp_loop.inc92:                                   ; preds = %omp.region.cont98
  %omp_loop.next97 = add nuw i64 %omp_loop.iv95, 1
  br label %omp_loop.header89

omp.loop_nest.region102:                          ; preds = %omp.loop_nest.region101
  br label %omp.loop_nest.region103

omp.loop_nest.region103:                          ; preds = %omp.loop_nest.region104, %omp.loop_nest.region102
  %29 = phi i64 [ %63, %omp.loop_nest.region104 ], [ 0, %omp.loop_nest.region102 ]
  %30 = icmp slt i64 %29, %4
  br i1 %30, label %omp.loop_nest.region104, label %omp.loop_nest.region105

omp.loop_nest.region105:                          ; preds = %omp.loop_nest.region103
  %31 = add i64 %27, 1
  br label %omp.loop_nest.region101

omp.loop_nest.region104:                          ; preds = %omp.loop_nest.region103
  %32 = mul nuw nsw i64 %25, %5
  %33 = mul nuw nsw i64 %27, %4
  %34 = add nuw nsw i64 %32, %33
  %35 = add nuw nsw i64 %34, %29
  %36 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %35
  %37 = load double, ptr %36, align 8
  %38 = add i64 %25, 2
  %39 = add i64 %27, 2
  %40 = mul nuw nsw i64 %38, %6
  %41 = mul nuw nsw i64 %39, %7
  %42 = add nuw nsw i64 %40, %41
  %43 = add nuw nsw i64 %42, %29
  %44 = getelementptr inbounds nuw double, ptr %loadgep_2, i64 %43
  %45 = load double, ptr %44, align 8
  %46 = add i64 %25, 1
  %47 = add i64 %27, 2
  %48 = mul nuw nsw i64 %46, %6
  %49 = mul nuw nsw i64 %47, %7
  %50 = add nuw nsw i64 %48, %49
  %51 = add nuw nsw i64 %50, %29
  %52 = getelementptr inbounds nuw double, ptr %loadgep_2, i64 %51
  %53 = load double, ptr %52, align 8
  %54 = fsub double %45, %53
  %55 = fmul double %37, %54
  %56 = fcmp ogt double %55, 0.000000e+00
  %57 = select i1 %56, double 0.000000e+00, double %37
  %58 = mul nuw nsw i64 %25, %8
  %59 = mul nuw nsw i64 %27, %4
  %60 = add nuw nsw i64 %58, %59
  %61 = add nuw nsw i64 %60, %29
  %62 = getelementptr inbounds nuw double, ptr %loadgep_4, i64 %61
  store double %57, ptr %62, align 8
  %63 = add i64 %29, 1
  br label %omp.loop_nest.region103

omp.par.exit76.exitStub:                          ; preds = %.fini121
  ret void
}

; Function Attrs: nounwind
define internal void @hdiff_fp64_polygeist..omp_par.1(ptr noalias %tid.addr21, ptr noalias %zero.addr22, ptr %0) #0 {
omp.par.entry23:
  %gep_.reloaded64 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded64 = load ptr, ptr %gep_.reloaded64, align 8, !align !1
  %gep_.reloaded65 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded65 = load ptr, ptr %gep_.reloaded65, align 8, !align !1
  %gep_.reloaded66 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded66 = load ptr, ptr %gep_.reloaded66, align 8, !align !1
  %gep_.reloaded67 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded67 = load ptr, ptr %gep_.reloaded67, align 8, !align !1
  %gep_.reloaded68 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
  %loadgep_.reloaded68 = load ptr, ptr %gep_.reloaded68, align 8, !align !1
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 5
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !2
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 6
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !2
  %p.lastiter58 = alloca i32, align 4
  %p.lowerbound59 = alloca i64, align 8
  %p.upperbound60 = alloca i64, align 8
  %p.stride61 = alloca i64, align 8
  %tid.addr.local27 = alloca i32, align 4
  %1 = load i32, ptr %tid.addr21, align 4
  store i32 %1, ptr %tid.addr.local27, align 4
  %tid28 = load i32, ptr %tid.addr.local27, align 4
  %2 = load i64, ptr %loadgep_.reloaded64, align 8
  %3 = load i64, ptr %loadgep_.reloaded65, align 8
  %4 = load i64, ptr %loadgep_.reloaded66, align 8
  %5 = load i64, ptr %loadgep_.reloaded67, align 8
  %6 = load i64, ptr %loadgep_.reloaded68, align 8
  br label %omp.region.after_alloca34

omp.region.after_alloca34:                        ; preds = %omp.par.entry23
  br label %omp.region.after_alloca31

omp.region.after_alloca31:                        ; preds = %omp.region.after_alloca34
  br label %omp.par.region24

omp.par.region24:                                 ; preds = %omp.region.after_alloca31
  br label %omp.par.region33

omp.par.region33:                                 ; preds = %omp.par.region24
  br label %omp.wsloop.region36

omp.wsloop.region36:                              ; preds = %omp.par.region33
  %7 = select i1 false, i64 %2, i64 0
  %8 = select i1 false, i64 0, i64 %2
  %9 = sub nsw i64 %8, %7
  %10 = icmp sle i64 %8, %7
  %11 = sub i64 %9, 1
  %12 = udiv i64 %11, 1
  %13 = add i64 %12, 1
  %14 = icmp ule i64 %9, 1
  %15 = select i1 %14, i64 1, i64 %13
  %omp_loop.tripcount37 = select i1 %10, i64 0, i64 %15
  br label %omp_loop.preheader38

omp_loop.preheader38:                             ; preds = %omp.wsloop.region36
  store i64 0, ptr %p.lowerbound59, align 8
  %16 = sub i64 %omp_loop.tripcount37, 1
  store i64 %16, ptr %p.upperbound60, align 8
  store i64 1, ptr %p.stride61, align 8
  %omp_global_thread_num62 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num62, i32 34, ptr %p.lastiter58, ptr %p.lowerbound59, ptr %p.upperbound60, ptr %p.stride61, i64 1, i64 0)
  %17 = load i64, ptr %p.lowerbound59, align 8
  %18 = load i64, ptr %p.upperbound60, align 8
  %19 = sub i64 %18, %17
  %20 = add i64 %19, 1
  br label %omp_loop.header39

omp_loop.header39:                                ; preds = %omp_loop.inc42, %omp_loop.preheader38
  %omp_loop.iv45 = phi i64 [ 0, %omp_loop.preheader38 ], [ %omp_loop.next47, %omp_loop.inc42 ]
  br label %omp_loop.cond40

omp_loop.cond40:                                  ; preds = %omp_loop.header39
  %omp_loop.cmp46 = icmp ult i64 %omp_loop.iv45, %20
  br i1 %omp_loop.cmp46, label %omp_loop.body41, label %omp_loop.exit43

omp_loop.exit43:                                  ; preds = %omp_loop.cond40
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num62)
  %omp_global_thread_num63 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num63)
  br label %omp_loop.after44

omp_loop.after44:                                 ; preds = %omp_loop.exit43
  br label %omp.region.cont35

omp.region.cont35:                                ; preds = %omp_loop.after44
  br label %omp.region.cont32

omp.region.cont32:                                ; preds = %omp.region.cont35
  br label %omp.par.pre_finalize25

omp.par.pre_finalize25:                           ; preds = %omp.region.cont32
  br label %.fini69

.fini69:                                          ; preds = %omp.par.pre_finalize25
  br label %omp.par.exit26.exitStub

omp_loop.body41:                                  ; preds = %omp_loop.cond40
  %21 = add i64 %omp_loop.iv45, %17
  %22 = mul i64 %21, 1
  %23 = add i64 %22, 0
  br label %omp.loop_nest.region49

omp.loop_nest.region49:                           ; preds = %omp_loop.body41
  %24 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region50

omp.loop_nest.region50:                           ; preds = %omp.loop_nest.region49
  br label %omp.loop_nest.region51

omp.loop_nest.region51:                           ; preds = %omp.loop_nest.region55, %omp.loop_nest.region50
  %25 = phi i64 [ %29, %omp.loop_nest.region55 ], [ 0, %omp.loop_nest.region50 ]
  %26 = icmp slt i64 %25, %3
  br i1 %26, label %omp.loop_nest.region52, label %omp.loop_nest.region56

omp.loop_nest.region56:                           ; preds = %omp.loop_nest.region51
  call void @llvm.stackrestore.p0(ptr %24)
  br label %omp.loop_nest.region57

omp.loop_nest.region57:                           ; preds = %omp.loop_nest.region56
  br label %omp.region.cont48

omp.region.cont48:                                ; preds = %omp.loop_nest.region57
  br label %omp_loop.inc42

omp_loop.inc42:                                   ; preds = %omp.region.cont48
  %omp_loop.next47 = add nuw i64 %omp_loop.iv45, 1
  br label %omp_loop.header39

omp.loop_nest.region52:                           ; preds = %omp.loop_nest.region51
  br label %omp.loop_nest.region53

omp.loop_nest.region53:                           ; preds = %omp.loop_nest.region54, %omp.loop_nest.region52
  %27 = phi i64 [ %51, %omp.loop_nest.region54 ], [ 0, %omp.loop_nest.region52 ]
  %28 = icmp slt i64 %27, %4
  br i1 %28, label %omp.loop_nest.region54, label %omp.loop_nest.region55

omp.loop_nest.region55:                           ; preds = %omp.loop_nest.region53
  %29 = add i64 %25, 1
  br label %omp.loop_nest.region51

omp.loop_nest.region54:                           ; preds = %omp.loop_nest.region53
  %30 = add i64 %23, 1
  %31 = add i64 %25, 1
  %32 = mul nuw nsw i64 %30, %5
  %33 = mul nuw nsw i64 %31, %4
  %34 = add nuw nsw i64 %32, %33
  %35 = add nuw nsw i64 %34, %27
  %36 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %35
  %37 = load double, ptr %36, align 8
  %38 = add i64 %25, 1
  %39 = mul nuw nsw i64 %23, %5
  %40 = mul nuw nsw i64 %38, %4
  %41 = add nuw nsw i64 %39, %40
  %42 = add nuw nsw i64 %41, %27
  %43 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %42
  %44 = load double, ptr %43, align 8
  %45 = fsub double %37, %44
  %46 = mul nuw nsw i64 %23, %6
  %47 = mul nuw nsw i64 %25, %4
  %48 = add nuw nsw i64 %46, %47
  %49 = add nuw nsw i64 %48, %27
  %50 = getelementptr inbounds nuw double, ptr %loadgep_2, i64 %49
  store double %45, ptr %50, align 8
  %51 = add i64 %27, 1
  br label %omp.loop_nest.region53

omp.par.exit26.exitStub:                          ; preds = %.fini69
  ret void
}

; Function Attrs: nounwind
define internal void @hdiff_fp64_polygeist..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #0 {
omp.par.entry:
  %gep_.reloaded = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded = load ptr, ptr %gep_.reloaded, align 8, !align !1
  %gep_.reloaded15 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded15 = load ptr, ptr %gep_.reloaded15, align 8, !align !1
  %gep_.reloaded16 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded16 = load ptr, ptr %gep_.reloaded16, align 8, !align !1
  %gep_.reloaded17 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded17 = load ptr, ptr %gep_.reloaded17, align 8, !align !1
  %gep_.reloaded18 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 4
  %loadgep_.reloaded18 = load ptr, ptr %gep_.reloaded18, align 8, !align !1
  %gep_.reloaded19 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 5
  %loadgep_.reloaded19 = load ptr, ptr %gep_.reloaded19, align 8, !align !1
  %gep_ = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 6
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !2
  %gep_1 = getelementptr { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 7
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
  %3 = load i64, ptr %loadgep_.reloaded15, align 8
  %4 = load i64, ptr %loadgep_.reloaded16, align 8
  %5 = load i64, ptr %loadgep_.reloaded17, align 8
  %6 = load i64, ptr %loadgep_.reloaded18, align 8
  %7 = load i64, ptr %loadgep_.reloaded19, align 8
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
  %8 = select i1 false, i64 %2, i64 0
  %9 = select i1 false, i64 0, i64 %2
  %10 = sub nsw i64 %9, %8
  %11 = icmp sle i64 %9, %8
  %12 = sub i64 %10, 1
  %13 = udiv i64 %12, 1
  %14 = add i64 %13, 1
  %15 = icmp ule i64 %10, 1
  %16 = select i1 %15, i64 1, i64 %14
  %omp_loop.tripcount = select i1 %11, i64 0, i64 %16
  br label %omp_loop.preheader

omp_loop.preheader:                               ; preds = %omp.wsloop.region
  store i64 0, ptr %p.lowerbound, align 8
  %17 = sub i64 %omp_loop.tripcount, 1
  store i64 %17, ptr %p.upperbound, align 8
  store i64 1, ptr %p.stride, align 8
  %omp_global_thread_num13 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num13, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i64 1, i64 0)
  %18 = load i64, ptr %p.lowerbound, align 8
  %19 = load i64, ptr %p.upperbound, align 8
  %20 = sub i64 %19, %18
  %21 = add i64 %20, 1
  br label %omp_loop.header

omp_loop.header:                                  ; preds = %omp_loop.inc, %omp_loop.preheader
  %omp_loop.iv = phi i64 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
  br label %omp_loop.cond

omp_loop.cond:                                    ; preds = %omp_loop.header
  %omp_loop.cmp = icmp ult i64 %omp_loop.iv, %21
  br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit

omp_loop.exit:                                    ; preds = %omp_loop.cond
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num13)
  %omp_global_thread_num14 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num14)
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
  %22 = add i64 %omp_loop.iv, %18
  %23 = mul i64 %22, 1
  %24 = add i64 %23, 0
  br label %omp.loop_nest.region

omp.loop_nest.region:                             ; preds = %omp_loop.body
  %25 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region5

omp.loop_nest.region5:                            ; preds = %omp.loop_nest.region
  %26 = add i64 %3, 2
  br label %omp.loop_nest.region6

omp.loop_nest.region6:                            ; preds = %omp.loop_nest.region10, %omp.loop_nest.region5
  %27 = phi i64 [ %31, %omp.loop_nest.region10 ], [ 0, %omp.loop_nest.region5 ]
  %28 = icmp slt i64 %27, %26
  br i1 %28, label %omp.loop_nest.region7, label %omp.loop_nest.region11

omp.loop_nest.region11:                           ; preds = %omp.loop_nest.region6
  call void @llvm.stackrestore.p0(ptr %25)
  br label %omp.loop_nest.region12

omp.loop_nest.region12:                           ; preds = %omp.loop_nest.region11
  br label %omp.region.cont4

omp.region.cont4:                                 ; preds = %omp.loop_nest.region12
  br label %omp_loop.inc

omp_loop.inc:                                     ; preds = %omp.region.cont4
  %omp_loop.next = add nuw i64 %omp_loop.iv, 1
  br label %omp_loop.header

omp.loop_nest.region7:                            ; preds = %omp.loop_nest.region6
  br label %omp.loop_nest.region8

omp.loop_nest.region8:                            ; preds = %omp.loop_nest.region9, %omp.loop_nest.region7
  %29 = phi i64 [ %80, %omp.loop_nest.region9 ], [ 0, %omp.loop_nest.region7 ]
  %30 = icmp slt i64 %29, %4
  br i1 %30, label %omp.loop_nest.region9, label %omp.loop_nest.region10

omp.loop_nest.region10:                           ; preds = %omp.loop_nest.region8
  %31 = add i64 %27, 1
  br label %omp.loop_nest.region6

omp.loop_nest.region9:                            ; preds = %omp.loop_nest.region8
  %32 = add i64 %24, 1
  %33 = add i64 %27, 1
  %34 = mul nuw nsw i64 %32, %5
  %35 = mul nuw nsw i64 %33, %6
  %36 = add nuw nsw i64 %34, %35
  %37 = add nuw nsw i64 %36, %29
  %38 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %37
  %39 = load double, ptr %38, align 8
  %40 = fmul double %39, 4.000000e+00
  %41 = add i64 %24, 2
  %42 = add i64 %27, 1
  %43 = mul nuw nsw i64 %41, %5
  %44 = mul nuw nsw i64 %42, %6
  %45 = add nuw nsw i64 %43, %44
  %46 = add nuw nsw i64 %45, %29
  %47 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %46
  %48 = load double, ptr %47, align 8
  %49 = add i64 %27, 1
  %50 = mul nuw nsw i64 %24, %5
  %51 = mul nuw nsw i64 %49, %6
  %52 = add nuw nsw i64 %50, %51
  %53 = add nuw nsw i64 %52, %29
  %54 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %53
  %55 = load double, ptr %54, align 8
  %56 = fadd double %48, %55
  %57 = add i64 %24, 1
  %58 = add i64 %27, 2
  %59 = mul nuw nsw i64 %57, %5
  %60 = mul nuw nsw i64 %58, %6
  %61 = add nuw nsw i64 %59, %60
  %62 = add nuw nsw i64 %61, %29
  %63 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %62
  %64 = load double, ptr %63, align 8
  %65 = fadd double %56, %64
  %66 = add i64 %24, 1
  %67 = mul nuw nsw i64 %66, %5
  %68 = mul nuw nsw i64 %27, %6
  %69 = add nuw nsw i64 %67, %68
  %70 = add nuw nsw i64 %69, %29
  %71 = getelementptr inbounds nuw double, ptr %loadgep_, i64 %70
  %72 = load double, ptr %71, align 8
  %73 = fadd double %65, %72
  %74 = fsub double %40, %73
  %75 = mul nuw nsw i64 %24, %7
  %76 = mul nuw nsw i64 %27, %4
  %77 = add nuw nsw i64 %75, %76
  %78 = add nuw nsw i64 %77, %29
  %79 = getelementptr inbounds nuw double, ptr %loadgep_2, i64 %78
  store double %74, ptr %79, align 8
  %80 = add i64 %29, 1
  br label %omp.loop_nest.region8

omp.par.exit.exitStub:                            ; preds = %.fini
  ret void
}

define void @_mlir_ciface_hdiff_fp64_polygeist(i64 %0, i64 %1, i64 %2, ptr %3, ptr %4, ptr %5) {
  %7 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %3, align 8
  %8 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 0
  %9 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 1
  %10 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 2
  %11 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 3, 0
  %12 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 3, 1
  %13 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 3, 2
  %14 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 4, 0
  %15 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 4, 1
  %16 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 4, 2
  %17 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %4, align 8
  %18 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %17, 0
  %19 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %17, 1
  %20 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %17, 2
  %21 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %17, 3, 0
  %22 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %17, 3, 1
  %23 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %17, 3, 2
  %24 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %17, 4, 0
  %25 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %17, 4, 1
  %26 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %17, 4, 2
  %27 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %5, align 8
  %28 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 0
  %29 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 1
  %30 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 2
  %31 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 3, 0
  %32 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 3, 1
  %33 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 3, 2
  %34 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 4, 0
  %35 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 4, 1
  %36 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 4, 2
  call void @hdiff_fp64_polygeist(i64 %0, i64 %1, i64 %2, ptr %8, ptr %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, ptr %18, ptr %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, ptr %28, ptr %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36)
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
  %22 = call i32 (ptr, ptr, ...) @fprintf(ptr %20, ptr @str7, i64 %5, ptr %21)
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
