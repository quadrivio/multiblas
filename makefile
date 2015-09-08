UNAME_S := $(shell uname -s)

HEADERS =   multiblas/src/multiblas_gemm.h \
            multiblas/src/gemm_r.h \
            multiblas/src/gemm_blas.h \
            multiblas/src/gemm_clblas.h \
            multiblas/src/gemm_opencl.h \
            multiblas/src/gemm_naive.h \
            multiblas/src/multiblas_crossprod.h \
            multiblas/src/crossprod_r.h \
            multiblas/src/crossprod_blas.h \
            multiblas/src/crossprod_clblas.h \
            multiblas/src/crossprod_opencl.h \
            multiblas/src/crossprod_naive.h \
            multiblas/src/multiblas.h \
            multiblas/src/opencl_info.h \
            multiblas/src/shim.h \
            multiblas/src/utils_clblas.h \
            multiblas/src/utils.h

multiblas:  obj \
            obj/main.o \
            obj/multiblas_gemm.o \
            obj/gemm_r.o \
            obj/gemm_blas.o \
            obj/gemm_clblas.o \
            obj/gemm_opencl.o \
            obj/gemm_naive.o \
            obj/multiblas_crossprod.o \
            obj/crossprod_r.o \
            obj/crossprod_blas.o \
            obj/crossprod_clblas.o \
            obj/crossprod_opencl.o \
            obj/crossprod_naive.o \
            obj/multiblas.o \
            obj/opencl_info.o \
            obj/shim.o \
            obj/utils_clblas.o \
            obj/utils.o \
            bin/inst/crossprod_f.cl \
            bin/inst/crossprod_d.cl \
            bin/inst/gemm_f.cl \
            bin/inst/gemm_d.cl
ifeq ($(UNAME_S),Darwin)
	g++ -o      bin/multiblas \
                obj/main.o \
                obj/multiblas_gemm.o \
                obj/gemm_r.o \
                obj/gemm_blas.o \
                obj/gemm_clblas.o \
                obj/gemm_opencl.o \
                obj/gemm_naive.o \
                obj/multiblas_crossprod.o \
                obj/crossprod_r.o \
                obj/crossprod_blas.o \
                obj/crossprod_clblas.o \
                obj/crossprod_opencl.o \
                obj/crossprod_naive.o \
                obj/multiblas.o \
                obj/opencl_info.o \
                obj/shim.o \
                obj/utils_clblas.o \
                obj/utils.o \
                -framework Accelerate \
                -framework OpenCL \
                -L/usr/local/clblas/lib/ -lclBLAS
else
	g++ -o      bin/multiblas \
                obj/main.o \
                obj/multiblas_gemm.o \
                obj/gemm_r.o \
                obj/gemm_blas.o \
                obj/gemm_clblas.o \
                obj/gemm_opencl.o \
                obj/gemm_naive.o \
                obj/multiblas_crossprod.o \
                obj/crossprod_r.o \
                obj/crossprod_blas.o \
                obj/crossprod_clblas.o \
                obj/crossprod_opencl.o \
                obj/crossprod_naive.o \
                obj/multiblas.o \
                obj/opencl_info.o \
                obj/shim.o \
                obj/utils_clblas.o \
                obj/utils.o \
                -L/usr/lib64/openblas/lib -lopenblas \
                -L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64 -lOpenCL \
                -L/opt/clBLAS-2.4.0-Linux-x64/lib64 -lclBLAS
endif

obj:
	mkdir -p obj

obj/main.o: multiBLAS.XC/multiBLAS.XC/main.cpp $(HEADERS)
	g++ -o obj/main.o -c multiBLAS.XC/multiBLAS.XC/main.cpp -Imultiblas/src \
    -I/opt/clBLAS-2.4.0-Linux-x64/include/ -I/usr/local/clblas/include -DRPACKAGE=0  -std=c++11

obj/multiblas_gemm.o: multiblas/src/multiblas_gemm.cpp $(HEADERS)
	g++ -o obj/multiblas_gemm.o -c multiblas/src/multiblas_gemm.cpp -Imultiblas/src \
    -I/opt/clBLAS-2.4.0-Linux-x64/include/ -I/usr/local/clblas/include -DRPACKAGE=0

obj/gemm_r.o: multiblas/src/gemm_r.cpp $(HEADERS)
	g++ -o obj/gemm_r.o -c multiblas/src/gemm_r.cpp -Imultiblas/src -DRPACKAGE=0

obj/gemm_blas.o: multiblas/src/gemm_blas.cpp $(HEADERS)
	g++ -o obj/gemm_blas.o -c multiblas/src/gemm_blas.cpp -Imultiblas/src -DRPACKAGE=0

obj/gemm_clblas.o: multiblas/src/gemm_clblas.cpp $(HEADERS)
	g++ -o obj/gemm_clblas.o -c multiblas/src/gemm_clblas.cpp -Imultiblas/src \
    -I/opt/clBLAS-2.4.0-Linux-x64/include/ -I/usr/local/clblas/include -DRPACKAGE=0

obj/gemm_opencl.o: multiblas/src/gemm_opencl.cpp $(HEADERS)
	g++ -o obj/gemm_opencl.o -c multiblas/src/gemm_opencl.cpp -Imultiblas/src -DRPACKAGE=0

obj/gemm_naive.o: multiblas/src/gemm_naive.cpp $(HEADERS)
	g++ -o obj/gemm_naive.o -c multiblas/src/gemm_naive.cpp -Imultiblas/src -DRPACKAGE=0

obj/multiblas_crossprod.o: multiblas/src/multiblas_crossprod.cpp $(HEADERS)
	g++ -o obj/multiblas_crossprod.o -c multiblas/src/multiblas_crossprod.cpp -Imultiblas/src \
    -I/opt/clBLAS-2.4.0-Linux-x64/include/ -I/usr/local/clblas/include -DRPACKAGE=0

obj/crossprod_r.o: multiblas/src/crossprod_r.cpp $(HEADERS)
	g++ -o obj/crossprod_r.o -c multiblas/src/crossprod_r.cpp -Imultiblas/src -DRPACKAGE=0

obj/crossprod_blas.o: multiblas/src/crossprod_blas.cpp $(HEADERS)
	g++ -o obj/crossprod_blas.o -c multiblas/src/crossprod_blas.cpp -Imultiblas/src -I/usr/lib64/openblas/include/ -DRPACKAGE=0

obj/crossprod_clblas.o: multiblas/src/crossprod_clblas.cpp $(HEADERS)
	g++ -o obj/crossprod_clblas.o -c multiblas/src/crossprod_clblas.cpp -Imultiblas/src \
    -I/opt/clBLAS-2.4.0-Linux-x64/include/ -I/usr/local/clblas/include -DRPACKAGE=0

obj/crossprod_opencl.o: multiblas/src/crossprod_opencl.cpp $(HEADERS)
	g++ -o obj/crossprod_opencl.o -c multiblas/src/crossprod_opencl.cpp -Imultiblas/src -I/opt/AMDAPPSDK-3.0-0-Beta/include -DRPACKAGE=0 -Dnullptr='NULL'

obj/crossprod_naive.o: multiblas/src/crossprod_naive.cpp $(HEADERS)
	g++ -o obj/crossprod_naive.o -c multiblas/src/crossprod_naive.cpp -Imultiblas/src -DRPACKAGE=0

obj/multiblas.o: multiblas/src/multiblas.cpp $(HEADERS)
	g++ -o obj/multiblas.o -c multiblas/src/multiblas.cpp -Imultiblas/src -DRPACKAGE=0

obj/opencl_info.o: multiblas/src/opencl_info.cpp $(HEADERS)
	g++ -o obj/opencl_info.o -c multiblas/src/opencl_info.cpp -Imultiblas/src -DRPACKAGE=0

obj/shim.o: multiblas/src/shim.cpp $(HEADERS)
	g++ -o obj/shim.o -c multiblas/src/shim.cpp -Imultiblas/src -DRPACKAGE=0

obj/utils_clblas.o: multiblas/src/utils_clblas.cpp $(HEADERS)
	g++ -o obj/utils_clblas.o -c multiblas/src/utils_clblas.cpp -Imultiblas/src \
    -I/opt/clBLAS-2.4.0-Linux-x64/include/ -I/usr/local/clblas/include -DRPACKAGE=0

obj/utils.o: multiblas/src/utils.cpp $(HEADERS)
	g++ -o obj/utils.o -c multiblas/src/utils.cpp -Imultiblas/src -DRPACKAGE=0

bin/inst/crossprod_f.cl: multiblas/inst/crossprod_f.cl
	mkdir -p bin/inst && cp multiblas/inst/crossprod_f.cl bin/inst/

bin/inst/crossprod_d.cl: multiblas/inst/crossprod_d.cl
	mkdir -p bin/inst && cp multiblas/inst/crossprod_d.cl bin/inst/

bin/inst/gemm_f.cl: multiblas/inst/gemm_f.cl
	mkdir -p bin/inst && cp multiblas/inst/gemm_f.cl bin/inst/

bin/inst/gemm_d.cl: multiblas/inst/gemm_d.cl
	mkdir -p bin/inst && cp multiblas/inst/gemm_d.cl bin/inst/

clean:
	rm -rf obj
	rm -rf bin
