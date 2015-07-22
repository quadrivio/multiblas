UNAME_S := $(shell uname -s)

HEADERS =   opencl.imp/src/crossprod_blas.h \
            opencl.imp/src/crossprod_clblas.h \
            opencl.imp/src/crossprod_opencl.h \
            opencl.imp/src/crossprod_naive.h \
            opencl.imp/src/opencl_info.h \
            opencl.imp/src/opencl_test.h \
            opencl.imp/src/shim.h \
            opencl.imp/src/opencl_imp.h \
            opencl.imp/src/utils.h

opencl_imp: obj \
            obj/main.o \
            obj/crossprod_blas.o \
            obj/crossprod_clblas.o \
            obj/crossprod_opencl.o \
            obj/crossprod_naive.o \
            obj/opencl_info.o \
            obj/opencl_test.o \
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
                obj/opencl_test.o \
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

obj/main.o: OpenCL.ImpXC/OpenCL.ImpXC/main.cpp $(HEADERS)
	g++ -o obj/main.o -c OpenCL.ImpXC/OpenCL.ImpXC/main.cpp -Iopencl.imp/src -DRPACKAGE=0  -std=c++11

obj/crossprod_blas.o: opencl.imp/src/crossprod_blas.cpp $(HEADERS)
	g++ -o obj/crossprod_blas.o -c opencl.imp/src/crossprod_blas.cpp -Iopencl.imp/src -I/usr/lib64/openblas/include/ -I../../clBLAS-master/src -DRPACKAGE=0

obj/crossprod_clblas.o: opencl.imp/src/crossprod_clblas.cpp $(HEADERS)
	g++ -o obj/crossprod_clblas.o -c opencl.imp/src/crossprod_clblas.cpp -Iopencl.imp/src -I/home/michael/clblas/clBLAS-2.2.0-Linux-x64/include/ -I../../clBLAS-master/src -DRPACKAGE=0

obj/crossprod_opencl.o: opencl.imp/src/crossprod_opencl.cpp $(HEADERS)
	g++ -o obj/crossprod_opencl.o -c opencl.imp/src/crossprod_opencl.cpp -Iopencl.imp/src -I/opt/AMDAPPSDK-3.0-0-Beta/include -DRPACKAGE=0 -Dnullptr='NULL'

obj/crossprod_naive.o: opencl.imp/src/crossprod_naive.cpp $(HEADERS)
	g++ -o obj/crossprod_naive.o -c opencl.imp/src/crossprod_naive.cpp -Iopencl.imp/src -DRPACKAGE=0

obj/opencl_info.o: opencl.imp/src/opencl_info.cpp $(HEADERS)
	g++ -o obj/opencl_info.o -c opencl.imp/src/opencl_info.cpp -Iopencl.imp/src -DRPACKAGE=0

obj/opencl_test.o: opencl.imp/src/opencl_test.cpp $(HEADERS)
	g++ -o obj/opencl_test.o -c opencl.imp/src/opencl_test.cpp -Iopencl.imp/src -DRPACKAGE=0 -Dnullptr='NULL'

obj/shim.o: opencl.imp/src/shim.cpp $(HEADERS)
	g++ -o obj/shim.o -c opencl.imp/src/shim.cpp -Iopencl.imp/src -DRPACKAGE=0

obj/opencl_imp.o: opencl.imp/src/opencl_imp.cpp $(HEADERS)
	g++ -o obj/opencl_imp.o -c opencl.imp/src/opencl_imp.cpp -Iopencl.imp/src -DRPACKAGE=0

obj/utils.o: opencl.imp/src/utils.cpp $(HEADERS)
	g++ -o obj/utils.o -c opencl.imp/src/utils.cpp -Iopencl.imp/src -DRPACKAGE=0

bin/inst/test_f.cl: opencl.imp/inst/test_f.cl
	mkdir -p bin/inst && cp opencl.imp/inst/test_f.cl bin/inst/

bin/inst/crossprod_f.cl: opencl.imp/inst/crossprod_f.cl
	mkdir -p bin/inst && cp opencl.imp/inst/crossprod_f.cl bin/inst/

bin/inst/crossprod_d.cl: opencl.imp/inst/crossprod_d.cl
	mkdir -p bin/inst && cp opencl.imp/inst/crossprod_d.cl bin/inst/

clean:
	rm -rf obj
	rm -rf bin
