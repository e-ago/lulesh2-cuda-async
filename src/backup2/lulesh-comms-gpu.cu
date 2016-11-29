
// If no MPI, then this whole file is stubbed out

#if USE_MPI
#include <mpi.h>
#include <string.h>
#include "comm.h"
#endif

#include "lulesh.h"

#if 0
#include "nvToolsExt.h"

#define COMM_VERT    1
#define COMM_HORIZ  2
#define SCAN_COL    3
#define APPEND_ROWS 4
#define ALL_REDUCE  5
#define EX_SCAN      6
#define SEND      7
#define RECEIVE      8
#define OTHER     9


#define PUSH_RANGE(name,cid)                                                                 \
   do {                                                                                                  \
     const uint32_t colors[] = {                                                             \
            0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff, 0xff000000, 0xff0000ff, 0x55ff3300, 0xff660000, 0x66330000  \
      };                                                                                                 \
      const int num_colors = sizeof(colors)/sizeof(colors[0]);                \
      int color_id = cid%num_colors;                                                   \
    nvtxEventAttributes_t eventAttrib = {0};                                  \
    eventAttrib.version = NVTX_VERSION;                                             \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                      \
    eventAttrib.colorType = NVTX_COLOR_ARGB;                                  \
    eventAttrib.color = colors[color_id];                                        \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                     \
    eventAttrib.message.ascii = name;                                               \
    nvtxRangePushEx(&eventAttrib);                                                  \
   } while(0)

#define POP_RANGE do { nvtxRangePop(); } while(0)
#endif
   
#if USE_MPI
/* Comm Routines */

#define ALLOW_UNPACKED_PLANE false
#define ALLOW_UNPACKED_ROW   false
#define ALLOW_UNPACKED_COL   false

/*
   There are coherence issues for packing and unpacking message
   buffers.  Ideally, you would like a lot of threads to 
   cooperate in the assembly/dissassembly of each message.
   To do that, each thread should really be operating in a
   different coherence zone.

   Let's assume we have three fields, f1 through f3, defined on
   a 61x61x61 cube.  If we want to send the block boundary
   information for each field to each neighbor processor across
   each cube face, then we have three cases for the
   memory layout/coherence of data on each of the six cube
   boundaries:

      (a) Two of the faces will be in contiguous memory blocks
      (b) Two of the faces will be comprised of pencils of
          contiguous memory.
      (c) Two of the faces will have large strides between
          every value living on the face.

   How do you pack and unpack this data in buffers to
   simultaneous achieve the best memory efficiency and
   the most thread independence?

   Do do you pack field f1 through f3 tighly to reduce message
   size?  Do you align each field on a cache coherence boundary
   within the message so that threads can pack and unpack each
   field independently?  For case (b), do you align each
   boundary pencil of each field separately?  This increases
   the message size, but could improve cache coherence so
   each pencil could be processed independently by a separate
   thread with no conflicts.

   Also, memory access for case (c) would best be done without
   going through the cache (the stride is so large it just causes
   a lot of useless cache evictions).  Is it worth creating
   a special case version of the packing algorithm that uses
   non-coherent load/store opcodes?
*/

/******************************************/

template<int type>
__global__ void SendPlane(Real_t *destAddr, Real_t *srcAddr, Index_t sendCount, Index_t dx, Index_t dy, Index_t dz)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= sendCount) return;

  int i, j;

  switch (type) {
  case 0:
    i = tid;
    destAddr[i] = srcAddr[i] ;
    break;
  case 1:
    i = tid;
    destAddr[i] = srcAddr[dx*dy*(dz - 1) + i] ;
    break;
  case 2:
    i = tid / dx;
    j = tid % dx;
    destAddr[i*dx+j] = srcAddr[i*dx*dy + j] ;
    break;
  case 3:
    i = tid / dx;
    j = tid % dx;
    destAddr[i*dx+j] = srcAddr[dx*(dy - 1) + i*dx*dy + j] ;
    break;
  case 4:
    i = tid / dy;
    j = tid % dy;
    destAddr[i*dy + j] = srcAddr[i*dx*dy + j*dx] ;
    break;
  case 5:
    i = tid / dy;
    j = tid % dy;
    destAddr[i*dy + j] = srcAddr[dx - 1 + i*dx*dy + j*dx] ;
    break;
  }
}

template<int type>
__global__ void AddPlane(Real_t *srcAddr, Real_t *destAddr, Index_t recvCount, Index_t dx, Index_t dy, Index_t dz)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= recvCount) return;

  int i, j;

  switch (type) {
  case 0:
    i = tid;
    destAddr[i] += srcAddr[i] ;
    break;
  case 1:
    i = tid;
    destAddr[dx*dy*(dz - 1) + i] += srcAddr[i] ;
    break;
  case 2:
    i = tid / dx;
    j = tid % dx;
    destAddr[i*dx*dy + j] += srcAddr[i*dx + j] ;
    break;
  case 3:
    i = tid / dx;
    j = tid % dx;
    destAddr[dx*(dy - 1) + i*dx*dy + j] += srcAddr[i*dx + j] ;
    break;
  case 4:
    i = tid / dy;
    j = tid % dy;
    destAddr[i*dx*dy + j*dx] += srcAddr[i*dy + j] ;
    break;
  case 5:
    i = tid / dy;
    j = tid % dy;
    destAddr[dx - 1 + i*dx*dy + j*dx] += srcAddr[i*dy + j] ;
    break;
  }
}

template<int type>
__global__ void CopyPlane(Real_t *srcAddr, Real_t *destAddr, Index_t recvCount, Index_t dx, Index_t dy, Index_t dz)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= recvCount) return;

  int i, j;

  switch (type) {
  case 0:
    i = tid;
    destAddr[i] = srcAddr[i] ;
    break;
  case 1:
    i = tid;
    destAddr[dx*dy*(dz - 1) + i] = srcAddr[i] ;
    break;
  case 2:
    i = tid / dx;
    j = tid % dx;
    destAddr[i*dx*dy + j] = srcAddr[i*dx + j] ;
    break;
  case 3:
    i = tid / dx;
    j = tid % dx;
    destAddr[dx*(dy - 1) + i*dx*dy + j] = srcAddr[i*dx + j] ;
    break;
  case 4:
    i = tid / dy;
    j = tid % dy;
    destAddr[i*dx*dy + j*dx] = srcAddr[i*dy + j] ;
    break;
  case 5:
    i = tid / dy;
    j = tid % dy;
    destAddr[dx - 1 + i*dx*dy + j*dx] = srcAddr[i*dy + j] ;
    break;
  }
}

template<int type>
__global__ void SendEdge(Real_t *destAddr, Real_t *srcAddr, Index_t sendCount, Index_t dx, Index_t dy, Index_t dz)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= sendCount) return;

  switch (type) {
  case 0:
    destAddr[i] = srcAddr[i*dx*dy] ;
    break;
  case 1:
    destAddr[i] = srcAddr[i] ;
    break;
  case 2:
    destAddr[i] = srcAddr[i*dx] ;
    break;
  case 3:
    destAddr[i] = srcAddr[dx*dy - 1 + i*dx*dy] ;
    break;
  case 4:
    destAddr[i] = srcAddr[dx*(dy-1) + dx*dy*(dz-1) + i] ;
    break;
  case 5:
    destAddr[i] = srcAddr[dx*dy*(dz-1) + dx - 1 + i*dx] ;
    break;
  case 6:
    destAddr[i] = srcAddr[dx*(dy-1) + i*dx*dy] ;
    break;
  case 7:
    destAddr[i] = srcAddr[dx*dy*(dz-1) + i] ;
    break;
  case 8:
    destAddr[i] = srcAddr[dx*dy*(dz-1) + i*dx] ;
    break;
  case 9:
    destAddr[i] = srcAddr[dx - 1 + i*dx*dy] ;
    break;
  case 10:
    destAddr[i] = srcAddr[dx*(dy - 1) + i] ;
    break;
  case 11:
    destAddr[i] = srcAddr[dx - 1 + i*dx] ;
    break;
  }
}

template<int type>
__global__ void AddEdge(Real_t *srcAddr, Real_t *destAddr, Index_t recvCount, Index_t dx, Index_t dy, Index_t dz)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= recvCount) return;

  switch (type) {
  case 0:
    destAddr[i*dx*dy] += srcAddr[i] ;
    break;
  case 1:
    destAddr[i] += srcAddr[i] ;
    break;
  case 2:
    destAddr[i*dx] += srcAddr[i] ;
    break;
  case 3:
    destAddr[dx*dy - 1 + i*dx*dy] += srcAddr[i] ;
    break;
  case 4:
    destAddr[dx*(dy-1) + dx*dy*(dz-1) + i] += srcAddr[i] ;
    break;
  case 5:
    destAddr[dx*dy*(dz-1) + dx - 1 + i*dx] += srcAddr[i] ;
    break;
  case 6:
    destAddr[dx*(dy-1) + i*dx*dy] += srcAddr[i] ;
    break;
  case 7:
    destAddr[dx*dy*(dz-1) + i] += srcAddr[i] ;
    break;
  case 8:
    destAddr[dx*dy*(dz-1) + i*dx] += srcAddr[i] ;
    break;
  case 9:
    destAddr[dx - 1 + i*dx*dy] += srcAddr[i] ;
    break;
  case 10:
    destAddr[dx*(dy - 1) + i] += srcAddr[i] ;
    break;
  case 11:
    destAddr[dx - 1 + i*dx] += srcAddr[i] ;
    break;
  }
}

template<int type>
__global__ void CopyEdge(Real_t *srcAddr, Real_t *destAddr, Index_t recvCount, Index_t dx, Index_t dy, Index_t dz)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= recvCount) return;

  switch (type) {
  case 0:
    destAddr[i*dx*dy] = srcAddr[i] ;
    break;
  case 1:
    destAddr[i] = srcAddr[i] ;
    break;
  case 2:
    destAddr[i*dx] = srcAddr[i] ;
    break;
  case 3:
    destAddr[dx*dy - 1 + i*dx*dy] = srcAddr[i] ;
    break;
  case 4:
    destAddr[dx*(dy-1) + dx*dy*(dz-1) + i] = srcAddr[i] ;
    break;
  case 5:
    destAddr[dx*dy*(dz-1) + dx - 1 + i*dx] = srcAddr[i] ;
    break;
  case 6:
    destAddr[dx*(dy-1) + i*dx*dy] = srcAddr[i] ;
    break;
  case 7:
    destAddr[dx*dy*(dz-1) + i] = srcAddr[i] ;
    break;
  case 8:
    destAddr[dx*dy*(dz-1) + i*dx] = srcAddr[i] ;
    break;
  case 9:
    destAddr[dx - 1 + i*dx*dy] = srcAddr[i] ;
    break;
  case 10:
    destAddr[dx*(dy - 1) + i] = srcAddr[i] ;
    break;
  case 11:
    destAddr[dx - 1 + i*dx] = srcAddr[i] ;
    break;
  }
}

__global__ void AddCorner(Real_t *destAddr, Real_t src)
{
  destAddr[0] += src;
}

__global__ void AddCornerAsync(Real_t *destAddr, Real_t * src)
{
  destAddr[0] += src[0];
}

__global__ void CopyCorner(Real_t *destAddr, Real_t src)
{
  destAddr[0] = src;
}

__global__ void CopyCornerAsync(Real_t *destAddr, Real_t * src)
{
  destAddr[0] = src[0];
}

/******************************************/

void CommSendGpu(Domain& domain, int msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz, bool doSend, bool planeOnly, cudaStream_t stream, int typeBuf)
{

   if (domain.numRanks() == 1)
      return ;

   /* post recieve buffers for all incoming messages */
   int myRank ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
   MPI_Datatype baseType = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE) ;
   MPI_Status status[26] ;
   Real_t *destAddr ;
   Real_t *d_destAddr ;
   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;
   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;
   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   for (Index_t i=0; i<26; ++i) {
      domain.sendRequest[i] = MPI_REQUEST_NULL ;
   }

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

   //if(myRank == 0)
   // //printf("\n***CommSendGpu***\n");

   // setup launch grid
   const int block = 128;

#if 0
   int sendRanks[26];
   for(int indexRank=0; indexRank<26; indexRank++)
    sendRanks[indexRank] = -1;
#endif

PUSH_RANGE("CommSendGpu", 1);
   int indexMsg=0;
   /* post sends */

   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dx * dy ;

      if (planeMin) {
        int toRank = myRank - domain.tp()*domain.tp();
	 destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm] ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendPlane<0><<<(sendCount+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), sendCount, dx, dy, dz);
            d_destAddr += sendCount ;
         }
         d_destAddr -= xferFields*sendCount ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*sendCount*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         indexMsg=pmsg;
         //sendRanks[indexMsg] = toRank;

         //if(myRank == 0) printf("---> 1 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
         comm_global_isend_stream(destAddr, xferFields*sendCount, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg], stream, indexMsg+(26*typeBuf));
         /*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank - domain.tp()*domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         */
         ++pmsg ;
      }
      if (planeMax && doSend) {
                int toRank = myRank + domain.tp()*domain.tp();


         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg] ;
	 d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm] ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	         SendPlane<1><<<(sendCount+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), sendCount, dx, dy, dz);
            d_destAddr += sendCount ;
         }
          d_destAddr -= xferFields*sendCount ;
          cudaMemcpyAsync(destAddr, d_destAddr, xferFields*sendCount*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
          
          indexMsg = pmsg;

          //if(myRank == 0) printf("---> 2 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
          //sendRanks[indexMsg] = toRank;
          comm_global_isend_stream(destAddr, xferFields*sendCount, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg], stream, indexMsg+(26*typeBuf)) ;

        /*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank + domain.tp()*domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
        */
         ++pmsg ;
      }
   }
   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dx * dz ;

      if (rowMin) {

        int toRank = myRank - domain.tp();

         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendPlane<2><<<(sendCount+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), sendCount, dx, dy, dz);
            d_destAddr += sendCount ;
         }
         d_destAddr -= xferFields*sendCount ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*sendCount*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         
         indexMsg = pmsg;
         //if(myRank == 0) printf("---> 3 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
         //sendRanks[indexMsg] = toRank;
         comm_global_isend_stream(destAddr, xferFields*sendCount, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);
         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank - domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
*/
         ++pmsg ;
      }
      if (rowMax && doSend) {
        int toRank = myRank + domain.tp();
         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendPlane<3><<<(sendCount+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), sendCount, dx, dy, dz);
            d_destAddr += sendCount ;
         }
         d_destAddr -= xferFields*sendCount ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*sendCount*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);

        indexMsg = pmsg;
        //if(myRank == 0) printf("---> 4 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*sendCount, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg], stream, indexMsg+(26*typeBuf));

/*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank + domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
*/
         ++pmsg ;
      }
   }
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dy * dz ;

      if (colMin) {
        int toRank = myRank - 1;
         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendPlane<4><<<(sendCount+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), sendCount, dx, dy, dz);
            d_destAddr += sendCount ;
         }
         d_destAddr -= xferFields*sendCount ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*sendCount*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
          
        indexMsg = pmsg;
        //if(myRank == 0) printf("---> 5 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*sendCount, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg], stream, indexMsg+(26*typeBuf));
        /*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank - 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
        */
        ++pmsg ;
      }
      if (colMax && doSend) {
         int toRank = myRank + 1 ;
         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendPlane<5><<<(sendCount+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), sendCount, dx, dy, dz);
            d_destAddr += sendCount ;
         }
         d_destAddr -= xferFields*sendCount ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*sendCount*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
        
        indexMsg = pmsg;
        //if(myRank == 0) printf("---> 6 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*sendCount, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank + 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
*/
        ++pmsg ;
      }
   }

   if (!planeOnly) {
      if (rowMin && colMin) {
         int toRank = myRank - domain.tp() - 1 ;
         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg + emsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<0><<<(dz+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dz, dx, dy, dz);
            d_destAddr += dz ;
         }
         d_destAddr -= xferFields*dz ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         
        indexMsg = pmsg+emsg;
        //if(myRank == 0) printf("---> 7 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*dz, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], stream, indexMsg+(26*typeBuf));

        /*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
        */

         ++emsg;
      }

      if (rowMin && planeMin) {
         int toRank = myRank - domain.tp()*domain.tp() - domain.tp() ;
         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg + emsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<1><<<(dx+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dx, dx, dy, dz);
            d_destAddr += dx ;
         }
         d_destAddr -= xferFields*dx ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         
        indexMsg = pmsg+emsg;
        //if(myRank == 0) printf("---> 8 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*dx, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
*/
        ++emsg ;
      }

      if (colMin && planeMin) {
         int toRank = myRank - domain.tp()*domain.tp() - 1 ;
         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg + emsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<2><<<(dy+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dy, dx, dy, dz);
            d_destAddr += dy ;
         }
         d_destAddr -= xferFields*dy ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         
        indexMsg = pmsg+emsg;
        //if(myRank == 0) printf("---> 9 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*dy, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], stream, indexMsg+(26*typeBuf));

/*
         cudaStreamSynchronize(stream);
         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
*/
         ++emsg ;
      }

      if (rowMax && colMax && doSend) {
         int toRank = myRank + domain.tp() + 1 ;
         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg + emsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<3><<<(dz+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dz, dx, dy, dz);
            d_destAddr += dz ;
         }
         d_destAddr -= xferFields*dz ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         
        indexMsg = pmsg+emsg;
        //if(myRank == 0) printf("---> 10 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*dz, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
*/
        ++emsg ;
      }

      if (rowMax && planeMax && doSend) {
         int toRank = myRank + domain.tp()*domain.tp() + domain.tp() ;
         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg + emsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<4><<<(dx+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dx, dx, dy, dz);
            d_destAddr += dx ;
         }
         d_destAddr -= xferFields*dx ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         
        indexMsg = pmsg+emsg;
        //if(myRank == 0) printf("---> 11 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*dx, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
*/
        ++emsg ;
      }

      if (colMax && planeMax && doSend) {
         int toRank = myRank + domain.tp()*domain.tp() + 1 ;
         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg + emsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<5><<<(dy+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dy, dx, dy, dz);
            d_destAddr += dy ;
         }
         d_destAddr -= xferFields*dy ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         
        indexMsg = pmsg+emsg;
        //if(myRank == 0) printf("---> 12 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*dy, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
*/
        ++emsg ;
      }

      if (rowMax && colMin && doSend) {
         int toRank = myRank + domain.tp() - 1 ;
         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg + emsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<6><<<(dz+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dz, dx, dy, dz);
            d_destAddr += dz ;
         }
         d_destAddr -= xferFields*dz ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         
        indexMsg = pmsg+emsg;
        //if(myRank == 0) printf("---> 13 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*dz, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], stream, indexMsg+(26*typeBuf));

/*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
*/
         ++emsg ;
      }

      if (rowMin && planeMax && doSend) {
        int toRank = myRank + domain.tp()*domain.tp() - domain.tp() ;
        destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg + emsg] ;
        d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm + emsg * maxEdgeComm] ;
        for (Index_t fi=0; fi<xferFields; ++fi) {
          Domain_member src = fieldData[fi] ;
          SendEdge<7><<<(dx+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dx, dx, dy, dz);
          d_destAddr += dx ;
        }
        d_destAddr -= xferFields*dx ;
        cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
        indexMsg = pmsg+emsg;
        //if(myRank == 0) printf("---> 14 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*dx, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
*/
         ++emsg ;
      }

      if (colMin && planeMax && doSend) {
         int toRank = myRank + domain.tp()*domain.tp() - 1 ;
         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg + emsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<8><<<(dy+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dy, dx, dy, dz);
            d_destAddr += dy ;
         }
         d_destAddr -= xferFields*dy ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         
        indexMsg = pmsg+emsg;
        //if(myRank == 0) printf("---> 15 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*dy, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], stream, indexMsg+(26*typeBuf));
        /*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
        ++emsg ;
      }

      if (rowMin && colMax) {
         int toRank = myRank - domain.tp() + 1 ;
         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg + emsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<9><<<(dz+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dz, dx, dy, dz);
            d_destAddr += dz ;
         }
         d_destAddr -= xferFields*dz ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         
        indexMsg = pmsg+emsg;
        //if(myRank == 0) printf("---> 16 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*dz, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
*/
        ++emsg ;
      }

      if (rowMax && planeMin) {
         int toRank = myRank - domain.tp()*domain.tp() + domain.tp() ;
         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg + emsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<10><<<(dx+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dx, dx, dy, dz);
            d_destAddr += dx ;
         }
         d_destAddr -= xferFields*dx ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         
        indexMsg = pmsg+emsg;
        //if(myRank == 0) printf("---> 17 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*dx, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], stream, indexMsg+(26*typeBuf));
        /*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
        ++emsg ;
      }

      if (colMax && planeMin) {
         int toRank = myRank - domain.tp()*domain.tp() + 1 ;
         destAddr = domain.commDataSendStream_multi[(26*typeBuf)+pmsg + emsg] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<11><<<(dy+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dy, dx, dy, dz);
            d_destAddr += dy ;
         }
         d_destAddr -= xferFields*dy ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         
        indexMsg = pmsg+emsg;
        //if(myRank == 0) printf("---> 18 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(destAddr, xferFields*dy, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], stream, indexMsg+(26*typeBuf));
        /*
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
        ++emsg ;
      }

      if (rowMin && colMin && planeMin) {
         /* corner at domain logical coord (0, 0, 0) */
         int toRank = myRank - domain.tp()*domain.tp() - domain.tp() - 1 ;
         Real_t *comBuf = domain.commDataSendStream_multi[(26*typeBuf)+pmsg+ emsg + cmsg];
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(0), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }

        indexMsg = pmsg+emsg+cmsg;
        //if(myRank == 0) printf("---> 19 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
*/
        ++cmsg ;
      }
      if (rowMin && colMin && planeMax && doSend) {
         /* corner at domain logical coord (0, 0, 1) */
         int toRank = myRank + domain.tp()*domain.tp() - domain.tp() - 1 ;
         Real_t *comBuf = domain.commDataSendStream_multi[(26*typeBuf)+pmsg+ emsg + cmsg] ;
         Index_t idx = dx*dy*(dz - 1) ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(idx), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }

        indexMsg = pmsg+emsg+cmsg;
        //if(myRank == 0) printf("---> 20 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
*/
        ++cmsg ;
      }
      if (rowMin && colMax && planeMin) {
         /* corner at domain logical coord (1, 0, 0) */
         int toRank = myRank - domain.tp()*domain.tp() - domain.tp() + 1 ;
         Real_t *comBuf = domain.commDataSendStream_multi[(26*typeBuf)+pmsg+ emsg + cmsg] ;
         Index_t idx = dx - 1 ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(idx), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }

        indexMsg = pmsg+emsg+cmsg;
        //if(myRank == 0) printf("---> 21 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
*/
         ++cmsg ;
      }
      if (rowMin && colMax && planeMax && doSend) {
         /* corner at domain logical coord (1, 0, 1) */
         int toRank = myRank + domain.tp()*domain.tp() - domain.tp() + 1 ;
         Real_t *comBuf = domain.commDataSendStream_multi[(26*typeBuf)+pmsg+ emsg + cmsg] ;
         Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(idx), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }

        indexMsg = pmsg+emsg+cmsg;
        //if(myRank == 0) printf("---> 22 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
*/
         ++cmsg ;
      }
      if (rowMax && colMin && planeMin) {
         /* corner at domain logical coord (0, 1, 0) */
         int toRank = myRank - domain.tp()*domain.tp() + domain.tp() - 1 ;
         Real_t *comBuf = domain.commDataSendStream_multi[(26*typeBuf)+pmsg+ emsg + cmsg] ;
         Index_t idx = dx*(dy - 1) ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(idx), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }

        indexMsg = pmsg+emsg+cmsg;
        //if(myRank == 0) printf("---> 23 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
*/
         ++cmsg ;
      }
      if (rowMax && colMin && planeMax && doSend) {
         /* corner at domain logical coord (0, 1, 1) */
         int toRank = myRank + domain.tp()*domain.tp() + domain.tp() - 1 ;
         Real_t *comBuf = domain.commDataSendStream_multi[(26*typeBuf)+pmsg+ emsg + cmsg] ;
         Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(idx), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }
        indexMsg = pmsg+emsg+cmsg;
        //if(myRank == 0) printf("---> 24 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
*/
         ++cmsg ;
      }
      if (rowMax && colMax && planeMin) {
         /* corner at domain logical coord (1, 1, 0) */
         int toRank = myRank - domain.tp()*domain.tp() + domain.tp() + 1 ;
         Real_t *comBuf = domain.commDataSendStream_multi[(26*typeBuf)+pmsg+ emsg + cmsg] ;
         Index_t idx = dx*dy - 1 ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(idx), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }
        indexMsg = pmsg+emsg+cmsg;
        //if(myRank == 0) printf("---> 25 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
*/
         ++cmsg ;
      }
      if (rowMax && colMax && planeMax && doSend) {
         /* corner at domain logical coord (1, 1, 1) */
         int toRank = myRank + domain.tp()*domain.tp() + domain.tp() + 1 ;
         Real_t *comBuf = domain.commDataSendStream_multi[(26*typeBuf)+pmsg+ emsg + cmsg] ;
         Index_t idx = dx*dy*dz - 1 ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(idx), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }

        indexMsg = pmsg+emsg+cmsg;
        //if(myRank == 0) printf("---> 26 - myRank: %d, indexMsg: %d typeBuf: %d dst: %d\n", myRank, indexMsg+(26*typeBuf), typeBuf, toRank);
        //sendRanks[indexMsg] = toRank;
        comm_global_isend_stream(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], stream, indexMsg+(26*typeBuf));
/*
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
*/
         ++cmsg ;
      }
   }




#if 0
  for(int index=0; index < 26; index++)
  {
    if(sendRanks[index] != -1)
      //printf("myRank %d, send to %d\n", myRank, sendRanks[index]);
  }  
#endif

  //cudaDeviceSynchronize();
  //if(myRank == 0)
  //  printf("***myRank : %d CommSendGpu END: %d sends***\n", myRank, pmsg+emsg+cmsg);


//if(pmsg+emsg+cmsg == 0)
//{
  
  for(int i=0; i<pmsg+emsg+cmsg; i++)
  {
    comm_global_wait_stream(&domain.sendRequest[i], &status[i], SEND_STREAM_REQUEST, stream, i);
    //cudaDeviceSynchronize();
    //  printf("\n***myRank: %d CommSendGpu END: %d sends, wait number: %d***\n", myRank, pmsg+emsg+cmsg, i);
  }

//}
/*
cudaDeviceSynchronize();

  printf("\n***2myRank: %d CommSendGpu END: %d sends***\n", myRank, pmsg+emsg+cmsg);

*/
   //MPI_Barrier(MPI_COMM_WORLD);
  POP_RANGE;
  //comm_global_wait_all_stream(MPI_Request *request, MPI_Status *status, int count, int type, cudaStream_t stream = NULL)

  //debug
  //cudaDeviceSynchronize();

  //MPI_Waitall(26, domain.sendRequest, status) ;
}

/******************************************/

void CommSBNGpu(Domain& domain, int xferFields, Domain_member *fieldData, cudaStream_t *streams, int typeBuf) {

   if (domain.numRanks() == 1)
      return ;

   /* summation order should be from smallest value to largest */
   /* or we could try out kahan summation! */

   int myRank ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
   Index_t dx = domain.sizeX + 1 ;
   Index_t dy = domain.sizeY + 1 ;
   Index_t dz = domain.sizeZ + 1 ;
   MPI_Status status ;
   Real_t *srcAddr ;
   Real_t *d_srcAddr ;
   Index_t rowMin, rowMax, colMin, colMax, planeMin, planeMax ;
   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = 1 ;
   if (domain.rowLoc() == 0) {
      rowMin = 0 ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = 0 ;
   }
   if (domain.colLoc() == 0) {
      colMin = 0 ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = 0 ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = 0 ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = 0 ;
   }

   // setup launch grid
   const int block = 128;

   // streams
   int s = 0;
   cudaStream_t stream;

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

 // if(myRank == 0)
  //  //printf("\n***CommSBNGpu***\n");
  
PUSH_RANGE("CommSBNGpu", 1);

   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dy ;

      if (planeMin) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
          comm_global_wait_stream(&domain.recvRequest[pmsg], &status, RECV_REQUEST, stream, pmsg);
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    AddPlane<0><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (planeMax) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
          comm_global_wait_stream(&domain.recvRequest[pmsg], &status, RECV_REQUEST, stream, pmsg);
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    AddPlane<1><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dz ;

      if (rowMin) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         comm_global_wait_stream(&domain.recvRequest[pmsg], &status, RECV_REQUEST, stream, pmsg);
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    AddPlane<2><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (rowMax) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         comm_global_wait_stream(&domain.recvRequest[pmsg], &status, RECV_REQUEST, stream, pmsg);
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    AddPlane<3><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dy * dz ;

      if (colMin) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         comm_global_wait_stream(&domain.recvRequest[pmsg], &status, RECV_REQUEST, stream, pmsg);
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    AddPlane<4><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (colMax) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         comm_global_wait_stream(&domain.recvRequest[pmsg], &status, RECV_REQUEST, stream, pmsg);
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    AddPlane<5><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin & colMin) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, stream, pmsg+emsg);
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<0><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMin & planeMin) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, stream, pmsg+emsg);
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<1><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMin & planeMin) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, stream, pmsg+emsg);
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<2><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMax & colMax) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, stream, pmsg+emsg);
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<3><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMax & planeMax) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, stream, pmsg+emsg);
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<4><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMax & planeMax) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, stream, pmsg+emsg);
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<5><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMax & colMin) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, stream, pmsg+emsg);
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<6><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMin & planeMax) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, stream, pmsg+emsg);
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<7><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMin & planeMax) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, stream, pmsg+emsg);
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<8><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMin & colMax) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, stream, pmsg+emsg);
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<9><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMax & planeMin) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, stream, pmsg+emsg);
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<10><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMax & planeMin) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, stream, pmsg+emsg);
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<11><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMin & colMin & planeMin) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 0, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(0), comBuf+fi);
      }
      ++cmsg ;
   }
   if (rowMin & colMin & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 0, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*(dz - 1) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf+fi) ;
      }
      ++cmsg ;
   }
   if (rowMin & colMax & planeMin) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 0, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx - 1 ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf+fi) ;
      }
      ++cmsg ;
   }
   if (rowMin & colMax & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 0, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf+fi) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMin & planeMin) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 1, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*(dy - 1) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf+fi) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMin & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 1, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf+fi) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMax & planeMin) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 1, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy - 1 ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf+fi) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMax & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 1, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*dz - 1 ;
      comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf+fi) ;
      }
      ++cmsg ;
   }

POP_RANGE;

    ////printf("\n***myRank: %d CommSBNGpu END: %d wait***\n", myRank, pmsg+emsg+cmsg);

   // don't need to call synchronize since it will be done automatically 
   // before kernels start to execute in NULL stream
   //debug
   //cudaDeviceSynchronize();
}

/******************************************/

void CommSyncPosVelGpu(Domain& domain, cudaStream_t *streams, int typeBuf) {

   if (domain.numRanks() == 1)
      return ;

   int myRank ;
   bool doRecv = false ;
   Index_t xferFields = 6 ; /* x, y, z, xd, yd, zd */
   Domain_member fieldData[6] ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
   Index_t dx = domain.sizeX + 1 ;
   Index_t dy = domain.sizeY + 1 ;
   Index_t dz = domain.sizeZ + 1 ;
   MPI_Status status ;
   Real_t *srcAddr ;
   Real_t *d_srcAddr ;
   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;

   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;
   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   fieldData[0] = &Domain::get_x ;
   fieldData[1] = &Domain::get_y ;
   fieldData[2] = &Domain::get_z ;
   fieldData[3] = &Domain::get_xd ;
   fieldData[4] = &Domain::get_yd ;
   fieldData[5] = &Domain::get_zd ;

   // setup launch grid
   const int block = 128;

   // streams
   int s = 0;
   cudaStream_t stream;

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
PUSH_RANGE("CommSyncPosVelGpu", 2);

      // if(myRank == 0)
   // //printf("\n***CommSyncPosVelGpu***\n");


   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dy ;

      if (planeMin && doRecv) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
               //if(myRank == 0) printf("---> WAIT 1 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
               comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
               //comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg);
         //MPI_it(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	     CopyPlane<0><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (planeMax) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
               //if(myRank == 0) printf("---> WAIT 2 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
               //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
               comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    CopyPlane<1><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dz ;

      if (rowMin && doRecv) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
               //if(myRank == 0) printf("---> WAIT 3 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
               //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
               comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    CopyPlane<2><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (rowMax) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
               //if(myRank == 0) printf("---> WAIT 4 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
               //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
               comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    CopyPlane<3><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dy * dz ;

      if (colMin && doRecv) {
         /* contiguous memory */
	 stream = streams[s++];
          srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
          d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
          //if(myRank == 0) printf("---> WAIT 5 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
          //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
          comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            CopyPlane<4><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (colMax) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
               //if(myRank == 0) printf("---> WAIT 6 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
               //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
               comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    CopyPlane<5><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin && colMin && doRecv) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
            //if(myRank == 0) printf("---> WAIT 7 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<0><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMin && planeMin && doRecv) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
            //if(myRank == 0) printf("---> WAIT 8 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<1><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMin && planeMin && doRecv) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
            //if(myRank == 0) printf("---> WAIT 9 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<2><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMax & colMax) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
            //if(myRank == 0) printf("---> WAIT 10 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<3><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMax & planeMax) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
            //if(myRank == 0) printf("---> WAIT 11 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<4><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMax & planeMax) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
            //if(myRank == 0) printf("---> WAIT 12 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<5><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMax & colMin) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
            //if(myRank == 0) printf("---> WAIT 13 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<6><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMin & planeMax) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
            //if(myRank == 0) printf("---> WAIT 14 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<7><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMin & planeMax) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
            //if(myRank == 0) printf("---> WAIT 15 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<8><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMin && colMax && doRecv) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
            //if(myRank == 0) printf("---> WAIT 16 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<9><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMax && planeMin && doRecv) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
            //if(myRank == 0) printf("---> WAIT 17 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<10><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMax && planeMin && doRecv) {
      stream = streams[s++];
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
            //if(myRank == 0) printf("---> WAIT 18 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<11><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMin & colMin & planeMin & doRecv) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 0, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
            //if(myRank == 0) printf("---> WAIT 19 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(0), comBuf+fi) ;
      }
      ++cmsg;
   }
   if (rowMin & colMin & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 0, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*(dz - 1) ;
            //if(myRank == 0) printf("---> WAIT 20 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf+fi) ;
      }
      ++cmsg ;
   }
   if (rowMin & colMax & planeMin & doRecv) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 0, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx - 1 ;
            //if(myRank == 0) printf("---> WAIT 21 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf+fi) ;
      }
      ++cmsg ;
   }
   if (rowMin & colMax & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 0, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
            //if(myRank == 0) printf("---> WAIT 22 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf+fi) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMin & planeMin & doRecv) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 1, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*(dy - 1) ;
            //if(myRank == 0) printf("---> WAIT 23 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf+fi) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMin & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 1, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
            //if(myRank == 0) printf("---> WAIT 24 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf+fi) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMax & planeMin & doRecv) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 1, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy - 1 ;
            //if(myRank == 0) printf("---> WAIT 25 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf+fi) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMax & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 1, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*dz - 1 ;
            //if(myRank == 0) printf("---> WAIT 26 - myRank: %d, numMsg: %d\n", myRank, pmsg+emsg+cmsg);
            //comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg);
            comm_global_wait_stream(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, stream, pmsg+emsg+cmsg);
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCornerAsync<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf+fi) ;
      }
      ++cmsg ;
   }

POP_RANGE;
   // don't need to call synchronize since it will be done automatically 
   // before kernels start to execute in NULL stream

    //printf("\n***myRank : %d CommSyncPosVelGpu END: %d wait***\n", myRank, pmsg+emsg+cmsg);
   //debug
   //cudaDeviceSynchronize();
}

/******************************************/

void CommMonoQGpu(Domain& domain, cudaStream_t stream, int typeBuf)
{
   if (domain.numRanks() == 1)
      return ;

   int myRank ;
   Index_t xferFields = 3 ; /* delv_xi, delv_eta, delv_zeta */
   Domain_member fieldData[3] ;
   Index_t fieldOffset[3] ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t dx = domain.sizeX ;
   Index_t dy = domain.sizeY ;
   Index_t dz = domain.sizeZ ;
   MPI_Status status ;
   Real_t *srcAddr ;
   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;
   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;
   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   /* point into ghost data area */
   // fieldData[0] = &(domain.delv_xi(domain.numElem())) ;
   // fieldData[1] = &(domain.delv_eta(domain.numElem())) ;
   // fieldData[2] = &(domain.delv_zeta(domain.numElem())) ;
   fieldData[0] = &Domain::get_delv_xi ;
   fieldData[1] = &Domain::get_delv_eta ;
   fieldData[2] = &Domain::get_delv_zeta ;
   fieldOffset[0] = domain.numElem ;
   fieldOffset[1] = domain.numElem ;
   fieldOffset[2] = domain.numElem ;

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;


PUSH_RANGE("CommMonoQGpu", 3);
//if(myRank == 0)
//    //printf("\n *** CommMonoQGpu start ***\n");

   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dy ;

      if (planeMin) {
        //printf("---> 1 - myRank: %d, pmsg: %d\n", myRank, pmsg);
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         comm_global_wait_stream(&domain.recvRequest[pmsg], &status, RECV_REQUEST, stream, pmsg);
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            cudaMemcpyAsync(&(domain.*dest)(fieldOffset[fi]), srcAddr, opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
      if (planeMax) {
        //printf("---> 2 - myRank: %d, pmsg: %d\n", myRank, pmsg);

         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         comm_global_wait_stream(&domain.recvRequest[pmsg], &status, RECV_REQUEST, stream, pmsg);
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            cudaMemcpyAsync(&(domain.*dest)(fieldOffset[fi]), srcAddr, opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dz ;

      if (rowMin) {


                //printf("---> 3 - myRank: %d, pmsg: %d\n", myRank, pmsg);


         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         comm_global_wait_stream(&domain.recvRequest[pmsg], &status, RECV_REQUEST, stream, pmsg);
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            cudaMemcpyAsync(&(domain.*dest)(fieldOffset[fi]), srcAddr, opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
      if (rowMax) {
                        //printf("---> 4 - myRank: %d, pmsg: %d\n", myRank, pmsg);

         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         comm_global_wait_stream(&domain.recvRequest[pmsg], &status, RECV_REQUEST, stream, pmsg);
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            cudaMemcpyAsync(&(domain.*dest)(fieldOffset[fi]), srcAddr, opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
   }
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dy * dz ;

      if (colMin) {
                        //printf("---> 5 - myRank: %d, pmsg: %d\n", myRank, pmsg);

         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         comm_global_wait_stream(&domain.recvRequest[pmsg], &status, RECV_REQUEST, stream, pmsg);
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            cudaMemcpyAsync(&(domain.*dest)(fieldOffset[fi]), srcAddr, opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
      if (colMax) {
          //printf("---> 6 - myRank: %d, pmsg: %d\n", myRank, pmsg);
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg] ;
         comm_global_wait_stream(&domain.recvRequest[pmsg], &status, RECV_REQUEST, stream, pmsg);
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            cudaMemcpyAsync(&(domain.*dest)(fieldOffset[fi]), srcAddr, opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   POP_RANGE;
   // don't need to call synchronize since it will be done automatically 
   // before kernels start to execute in NULL stream
    //printf("\n***myRank : %d CommMonoQGpu END: %d wait***\n", myRank, pmsg);

   //debug
   //cudaDeviceSynchronize();
}

#endif
