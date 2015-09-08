UNAME_S := $(shell uname -s)

HEADERS =   multiblas/src/crossprod_blas.h \
            multiblas/src/crossprod_clblas.h \
            multiblas/src/crossprod_opencl.h \
            multiblas/src/crossprod_naive.h \
            multiblas/src/opencl_info.h \
            multiblas/src/opencl_test.h \
            multiblas/src/shim.h \
            multiblas/src/opencl_imp.h \
            multiblas/src/utils.h

opencl_imp: obj \
            obj/main.o \
            obj/crossprod_blas.o \
            obj/crossprod_clblas.o \
            obj/crossprod_opencl.o \
            obj/crossprod_naive.o \
            obj/opencl_info.o \
            obj/shim.o \
            obj/opencl_imp.o \
            obj/crossprod_opencl.o obj/utils.o \
            bin/inst/test_f.cl \
            bin/inst/crossprod_f.cl \
            bin/inst/crossprod_d.cl
ifeq ($(UNAME_S),Darwin)
	g++ -o      bin/opencl_imp \
                obj/main.o \
                obj/crossprod_blas.o \
                obj/crossprod_clblas.o \
                obj/crossprod_opencl.o \
                obj/crossprod_naive.o \
                obj/opencl_info.o \
                obj/shim.o \
                obj/opencl_imp.o \
                obj/utils.o \
                -framework Accelerate \
                -framework OpenCL \
                -L../../clBLAS-master/library/ -lclBLAS
else
	g++ -o      bin/opencl_imp \
                obj/main.o \
                obj/crossprod_blas.o \
                obj/crossprod_clblas.o \
                obj/crossprod_opencl.o \
                obj/crossprod_naive.o \
                obj/opencl_info.o \
                obj/opencl_test.o \
                obj/shim.o \
                obj/opencl_imp.o \
                obj/utils.o \
                -L/usr/lib64/openblas/lib -lopenblas \
                -L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64 -lOpenCL \
                -L/home/michael/clblas/clBLAS-2.2.0-Linux-x64/lib64 -lclBLAS
endif

obj:
	mkdir -p obj

obj/main.o: multiBLAS.XC/multiBLAS.XC/main.cpp $(HEADERS)
	g++ -o obj/main.o -c multiBLAS.XC/multiBLAS.XC/main.cpp -Imultiblas/src -DRPACKAGE=0  -std=c++11

obj/crossprod_blas.o: multiblas/src/crossprod_blas.cpp $(HEADERS)
	g++ -o obj/crossprod_blas.o -c multiblas/src/crossprod_blas.cpp -Imultiblas/src -I/usr/lib64/openblas/include/ -I../../clBLAS-master/src -DRPACKAGE=0

obj/crossprod_clblas.o: multiblas/src/crossprod_clblas.cpp $(HEADERS)
	g++ -o obj/crossprod_clblas.o -c multiblas/src/crossprod_clblas.cpp -Imultiblas/src -I/home/michael/clblas/clBLAS-2.2.0-Linux-x64/include/ -I../../clBLAS-master/src -DRPACKAGE=0

obj/crossprod_opencl.o: multiblas/src/crossprod_opencl.cpp $(HEADERS)
	g++ -o obj/crossprod_opencl.o -c multiblas/src/crossprod_opencl.cpp -Imultiblas/src -I/opt/AMDAPPSDK-3.0-0-Beta/include -DRPACKAGE=0 -Dnullptr='NULL'

obj/crossprod_naive.o: multiblas/src/crossprod_naive.cpp $(HEADERS)
	g++ -o obj/crossprod_naive.o -c multiblas/src/crossprod_naive.cpp -Imultiblas/src -DRPACKAGE=0

obj/opencl_info.o: multiblas/src/opencl_info.cpp $(HEADERS)
	g++ -o obj/opencl_info.o -c multiblas/src/opencl_info.cpp -Imultiblas/src -DRPACKAGE=0

obj/shim.o: multiblas/src/shim.cpp $(HEADERS)
	g++ -o obj/shim.o -c multiblas/src/shim.cpp -Imultiblas/src -DRPACKAGE=0

obj/opencl_imp.o: multiblas/src/opencl_imp.cpp $(HEADERS)
	g++ -o obj/opencl_imp.o -c multiblas/src/opencl_imp.cpp -Imultiblas/src -DRPACKAGE=0

obj/utils.o: multiblas/src/utils.cpp $(HEADERS)
	g++ -o obj/utils.o -c multiblas/src/utils.cpp -Imultiblas/src -DRPACKAGE=0

bin/inst/test_f.cl: multiblas/inst/test_f.cl
	mkdir -p bin/inst && cp multiblas/inst/test_f.cl bin/inst/

bin/inst/crossprod_f.cl: multiblas/inst/crossprod_f.cl
	mkdir -p bin/inst && cp multiblas/inst/crossprod_f.cl bin/inst/

bin/inst/crossprod_d.cl: multiblas/inst/crossprod_d.cl
	mkdir -p bin/inst && cp multiblas/inst/crossprod_d.cl bin/inst/

clean:
	rm -rf obj
	rm -rf bin
