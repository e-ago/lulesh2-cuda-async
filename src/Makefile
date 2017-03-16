CC=g++
# -m64
LD=nvcc 
#-m64

MPI_INCLUDE= -I$(MPI_HOME)/include
MP_INCLUDE= -I$(PREFIX)/include
MP_LIB= -L$(PREFIX)/lib -lmp -lgdsync -lgdrapi -libverbs 

CFLAGS= -c -O3 -DNDEBUG -DUSE_MPI
LDFLAGS= -lm -L$(MPI_HOME)/lib -lmpi_cxx -lmpi ${MP_LIB}
CUDACC=$(CUDA_HOME)/bin/nvcc

CUDACFLAGS= -c -O3 -arch=sm_35 -DUSE_MPI -lineinfo
#-Xptxas=-dlcm=ca
CUDAPTXFLAGS= -ptx -O3 
CUDADIR=$(CUDA_HOME)
CUDAINCLUDEDIR= $(CUDADIR)/include
CUDALDFLAGS=-L$(CUDADIR)/lib64  -lcudart -arch=sm_35 -lcuda  -lnvToolsExt
#-L$(CUDASDKLIB)
all: lulesh

lulesh: allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o comm.o
		${LD} -o lulesh allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o comm.o ${LDFLAGS} ${CUDALDFLAGS}

#hsgmpi.o: hsgmpi.c
#		mpic++ -c -m64 hsgmpi.c -DMYREAL=float -DMYREALENE=double -DNCOMP=3 ${MP_INCLUDE}

#assign_process_gpu.o: assign_process_gpu.c
#		mpicc -c -m64 assign_process_gpu.c -DMYREAL=float -DMYREALENE=double -DNCOMP=3 -I$(CUDA_HOME)/include
        
#.c.o:; $(CC) $(CFLAGS) $< -o $@

allocator.o: allocator.cu vector.h
	$(CUDACC) $(CUDACFLAGS) -I$(CUDAINCLUDEDIR) -I./ allocator.cu

lulesh.o: lulesh.cu util.h vector.h allocator.h
	$(CUDACC) $(CUDACFLAGS) -I$(CUDAINCLUDEDIR) -I./ ${MPI_INCLUDE} ${MP_INCLUDE} lulesh.cu

lulesh-comms.o: lulesh-comms.cu
	$(CUDACC) $(CUDACFLAGS) -I$(CUDAINCLUDEDIR) -I./ ${MPI_INCLUDE} ${MP_INCLUDE} lulesh-comms.cu

lulesh-comms-gpu.o: lulesh-comms-gpu.cu
	$(CUDACC) $(CUDACFLAGS) -I$(CUDAINCLUDEDIR) -I./ ${MPI_INCLUDE} ${MP_INCLUDE} lulesh-comms-gpu.cu

#	$(NVCC) $(FLAGS) allocator.cu -I ./ -c -o allocator.o

.cc.o:; 
	$(CC) $(CFLAGS) ${MPI_INCLUDE} ${MP_INCLUDE} -I$(CUDAINCLUDEDIR) $< -o $@

#.cu.o:;
#	$(CUDACC) $(CFLAGS) $(CUDACFLAGS) -I$(CUDAINCLUDEDIR) $< -o $@

#%.ptx: %.cu
#		$(CUDACC) $(CUDAPTXFLAGS) -I$(CUDAINCLUDEDIR) $(@:.ptx=.cu) -o $(@:.o=.ptx)

#%.cubin: %.cu
#		$(CUDACC) $(CUDACUBINFLAGS) -I$(CUDAINCLUDEDIR)  $(@:.cubin=.cu) -o $(@:.o=.cubin)

#redusum.o: redusum.cu
#		$(CUDACC) $(CUDACFLAGS) -I$(CUDAINCLUDEDIR) redusum.cu

clean:
		rm -rf *.o lulesh xyz.asc regNumList.txt
