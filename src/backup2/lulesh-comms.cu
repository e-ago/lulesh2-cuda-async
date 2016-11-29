
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


/* doRecv flag only works with regular block structure */
void CommRecv(Domain& domain, int msgType, Index_t xferFields,
              Index_t dx, Index_t dy, Index_t dz, bool doRecv, bool planeOnly, int typeBuf) {

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
      domain.recvRequest[i] = MPI_REQUEST_NULL ;
   }

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;


#if 0
   int recvRanks[26];
   for(int indexRank=0; indexRank<26; indexRank++)
    recvRanks[indexRank] = -1;
#endif

  PUSH_RANGE("CommRecv", 3);

   /* post receives */
   ////if(myRank == 0)
    ////printf("\n *** CommRecv start ***\n");
   /* receive data from neighboring domain faces */
   if (planeMin && doRecv) {
      /* contiguous memory */
      int fromRank = myRank - domain.tp()*domain.tp() ;
      int recvCount = dx * dy * xferFields ;

      //recvRanks[pmsg+emsg+cmsg]=fromRank;

      comm_global_irecv(domain.commDataRecv_multi[pmsg+(26*typeBuf)],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg], pmsg+(26*typeBuf), NULL); // domain.streams[2]);
/*
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]);
*/
      ++pmsg ;
   }
   if (planeMax) {
      /* contiguous memory */
      int fromRank = myRank + domain.tp()*domain.tp() ;
      int recvCount = dx * dy * xferFields ;

      //recvRanks[pmsg+emsg+cmsg]=fromRank;

      comm_global_irecv(domain.commDataRecv_multi[pmsg+(26*typeBuf)],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg], pmsg+(26*typeBuf), NULL);
      /*
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      */
      ++pmsg ;
   }
   if (rowMin && doRecv) {
      /* semi-contiguous memory */
      int fromRank = myRank - domain.tp() ;
      int recvCount = dx * dz * xferFields ;

      //recvRanks[pmsg+emsg+cmsg]=fromRank;

      comm_global_irecv(domain.commDataRecv_multi[pmsg+(26*typeBuf)],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg], pmsg+(26*typeBuf), NULL);
      /*
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      */
      ++pmsg ;
   }
   if (rowMax) {
      /* semi-contiguous memory */
      int fromRank = myRank + domain.tp() ;
      int recvCount = dx * dz * xferFields ;

      //recvRanks[pmsg+emsg+cmsg]=fromRank;

      comm_global_irecv(domain.commDataRecv_multi[pmsg+(26*typeBuf)],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg], pmsg+(26*typeBuf), NULL);
      /*
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      
      */
      ++pmsg ;
   }
   if(colMin && doRecv) {
      /* scattered memory */
      int fromRank = myRank - 1 ;
      int recvCount = dy * dz * xferFields ;

      //recvRanks[pmsg+emsg+cmsg]=fromRank;

      comm_global_irecv(domain.commDataRecv_multi[pmsg+(26*typeBuf)],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg], pmsg+(26*typeBuf), NULL);
      /*
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
      */
      ++pmsg ;
   }
   if (colMax) {
      /* scattered memory */
      int fromRank = myRank + 1 ;
      int recvCount = dy * dz * xferFields ;

      //recvRanks[pmsg+emsg+cmsg]=fromRank;

      comm_global_irecv(domain.commDataRecv_multi[pmsg+(26*typeBuf)],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg], pmsg+(26*typeBuf), NULL);
      /*
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
         */
            ++pmsg;

            }

   if (!planeOnly) {
      /* receive data from domains connected only by an edge */
      if (rowMin && colMin && doRecv) {
        int fromRank = myRank - domain.tp() - 1 ;
        int recvCount = dz * xferFields ;
        //recvRanks[pmsg+emsg+cmsg]=fromRank;

        comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf), NULL);
         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dz * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
          */
            ++emsg ;

             }

      if (rowMin && planeMin && doRecv) {
         int fromRank = myRank - domain.tp()*domain.tp() - domain.tp() ;
        int recvCount = dx * xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf), NULL);
         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dx * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         */
            ++emsg ;

             }

      if (colMin && planeMin && doRecv) {
         int fromRank = myRank - domain.tp()*domain.tp() - 1 ;
         int recvCount = dy * xferFields;
        //recvRanks[pmsg+emsg+cmsg]=fromRank;

        comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dy * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         */
            ++emsg ;

             }

      if (rowMax && colMax) {
         int fromRank = myRank + domain.tp() + 1 ;
         int recvCount = dz * xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dz * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         */
            ++emsg ;

             }

      if (rowMax && planeMax) {
         int fromRank = myRank + domain.tp()*domain.tp() + domain.tp() ;
        int recvCount = dx * xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dx * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         */
            ++emsg ;

             }

      if (colMax && planeMax) {
         int fromRank = myRank + domain.tp()*domain.tp() + 1 ;
          int recvCount = dy * xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dy * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         */
            ++emsg ;

             }

      if (rowMax && colMin) {
         int fromRank = myRank + domain.tp() - 1 ;
          int recvCount = dz * xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dz * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         */
            ++emsg ;

             }

      if (rowMin && planeMax) {
         int fromRank = myRank + domain.tp()*domain.tp() - domain.tp() ;
        int recvCount = dx * xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dx * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         */
            ++emsg ;

             }

      if (colMin && planeMax) {
         int fromRank = myRank + domain.tp()*domain.tp() - 1 ;
        int recvCount = dy * xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dy * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         */
            ++emsg ;

             }

      if (rowMin && colMax && doRecv) {
         int fromRank = myRank - domain.tp() + 1 ;
        int recvCount = dz * xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dz * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         */
            ++emsg ;

             }

      if (rowMax && planeMin && doRecv) {
         int fromRank = myRank - domain.tp()*domain.tp() + domain.tp() ;
         int recvCount = dx * xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dx * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         */
            ++emsg ;

             }

      if (colMax && planeMin && doRecv) {
         int fromRank = myRank - domain.tp()*domain.tp() + 1 ;
         int recvCount = dy * xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm],
                   dy * xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
         
        */
        ++emsg ;
             }

      /* receive data from domains connected only by a corner */
      if (rowMin && colMin && planeMin && doRecv) {
         /* corner at domain logical coord (0, 0, 0) */
         int fromRank = myRank - domain.tp()*domain.tp() - domain.tp() - 1 ;
        int recvCount = xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         */
        ++cmsg ;
      }
      if (rowMin && colMin && planeMax) {
         /* corner at domain logical coord (0, 0, 1) */
         int fromRank = myRank + domain.tp()*domain.tp() - domain.tp() - 1 ;
        int recvCount = xferFields;

         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         */
        ++cmsg ;
      }
      if (rowMin && colMax && planeMin && doRecv) {
         /* corner at domain logical coord (1, 0, 0) */
         int fromRank = myRank - domain.tp()*domain.tp() - domain.tp() + 1 ;
                 int recvCount = xferFields;

         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         */
        ++cmsg ;
      }
      if (rowMin && colMax && planeMax) {
         /* corner at domain logical coord (1, 0, 1) */
         int fromRank = myRank + domain.tp()*domain.tp() - domain.tp() + 1 ;
                 int recvCount = xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         */
        ++cmsg ;
      }
      if (rowMax && colMin && planeMin && doRecv) {
         /* corner at domain logical coord (0, 1, 0) */
         int fromRank = myRank - domain.tp()*domain.tp() + domain.tp() - 1 ;
                 int recvCount = xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         */
        ++cmsg ;
      }
      if (rowMax && colMin && planeMax) {
         /* corner at domain logical coord (0, 1, 1) */
         int fromRank = myRank + domain.tp()*domain.tp() + domain.tp() - 1 ;
                 int recvCount = xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf), NULL);

         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         */
        ++cmsg ;
      }
      if (rowMax && colMax && planeMin && doRecv) {
         /* corner at domain logical coord (1, 1, 0) */
         int fromRank = myRank - domain.tp()*domain.tp() + domain.tp() + 1 ;
                 int recvCount = xferFields;
         //recvRanks[pmsg+emsg+cmsg]=fromRank;

         comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg],
             recvCount, baseType, fromRank, msgType,
              MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf), NULL);
         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         */
        ++cmsg ;
      }
      if (rowMax && colMax && planeMax) {
         /* corner at domain logical coord (1, 1, 1) */
        int fromRank = myRank + domain.tp()*domain.tp() + domain.tp() + 1 ;
        int recvCount = xferFields;
        //recvRanks[pmsg+emsg+cmsg]=fromRank;

        comm_global_irecv(domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg],
            recvCount, baseType, fromRank, msgType,
            MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf), NULL);
         /*
         MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
                                         emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL],
                   xferFields, baseType, fromRank, msgType,
                   MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
         */
        ++cmsg ;
      }
   }

   //if(myRank == 0)
   //printf("\n***rank %d CommRecv END: %d recv***\n", myRank, pmsg+emsg+cmsg);

#if 0
  
    for(int index=0; index < 26; index++)
  {
    if(recvRanks[index] != -1)
      //printf("myRank %d, recv from %d\n", myRank, recvRanks[index]);
  }  
#endif

  //MPI_Barrier(MPI_COMM_WORLD);

  POP_RANGE;

}

/******************************************/

void CommSend(Domain& domain, int msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz, bool doSend, bool planeOnly, int typeBuf)
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
    //printf("\n *** CommSend start ***\n");
   /* post sends */

   PUSH_RANGE("CommSend", 2);

   if (planeMin | planeMax) {

      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dx * dy ;
      int toRank = myRank - domain.tp()*domain.tp();
      if (planeMin) {

            //if(myRank == 0) //printf("1\n");

         destAddr = domain.commDataSend_multi[pmsg+(26*typeBuf)] ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<sendCount; ++i) {
               destAddr[i] = (domain.*src)(i) ;
            }
            destAddr += sendCount ;
         }
         destAddr -= xferFields*sendCount ;

        comm_global_isend(destAddr, xferFields*sendCount, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg], pmsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank - domain.tp()*domain.tp(), msgType,
                            MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         */
         ++pmsg ;
      }
      if (planeMax && doSend) {
            //if(myRank == 0) //printf("2\n");

         destAddr = domain.commDataSend_multi[pmsg+(26*typeBuf)] ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<sendCount; ++i) {
               destAddr[i] = (domain.*src)(dx*dy*(dz - 1) + i) ;
            }
            destAddr += sendCount ;
         }
         destAddr -= xferFields*sendCount ;

comm_global_isend(destAddr, xferFields*sendCount, baseType,
                   myRank + domain.tp()*domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg], pmsg+(26*typeBuf)) ;
         /*
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
                //if(myRank == 0) //printf("3\n");

         destAddr = domain.commDataSend_multi[pmsg+(26*typeBuf)] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dx; ++j) {
                  destAddr[i*dx+j] = (domain.*src)(i*dx*dy + j) ;
               }
            }
            destAddr += sendCount ;
         }
         destAddr -= xferFields*sendCount ;

comm_global_isend(destAddr, xferFields*sendCount, baseType,
                   myRank - domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg], pmsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank - domain.tp(), msgType,
                            MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         */
         ++pmsg ;
      }
      if (rowMax && doSend) {
            //if(myRank == 0) //printf("4\n");

         destAddr = domain.commDataSend_multi[pmsg+(26*typeBuf)] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dx; ++j) {
                  destAddr[i*dx+j] = (domain.*src)(dx*(dy - 1) + i*dx*dy + j) ;
               }
            }
            destAddr += sendCount ;
         }
         destAddr -= xferFields*sendCount ;

comm_global_isend(destAddr, xferFields*sendCount, baseType,
                   myRank + domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg], pmsg+(26*typeBuf)) ;
         /*
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
                //if(myRank == 0) //printf("5\n");

         destAddr = domain.commDataSend_multi[pmsg+(26*typeBuf)] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  destAddr[i*dy + j] = (domain.*src)(i*dx*dy + j*dx) ;
               }
            }
            destAddr += sendCount ;
         }
         destAddr -= xferFields*sendCount ;

comm_global_isend(destAddr, xferFields*sendCount, baseType,
                   myRank - 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg], pmsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank - 1, msgType,
                            MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         */
         ++pmsg ;
      }
      if (colMax && doSend) {

            //if(myRank == 0) //printf("6\n");


         destAddr = domain.commDataSend_multi[pmsg+(26*typeBuf)] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  destAddr[i*dy + j] = (domain.*src)(dx - 1 + i*dx*dy + j*dx) ;
               }
            }
            destAddr += sendCount ;
         }
         destAddr -= xferFields*sendCount ;

comm_global_isend(destAddr, xferFields*sendCount, baseType,
                   myRank + 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg], pmsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank + 1, msgType,
                            MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         */
         ++pmsg ;
      }
   }

   if (!planeOnly) {
      if (rowMin && colMin) {

            //if(myRank == 0) //printf("7\n");


         int toRank = myRank - domain.tp() - 1 ;
         destAddr = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               destAddr[i] = (domain.*src)(i*dx*dy) ;
            }
            destAddr += dz ;
         }
         destAddr -= xferFields*dz ;
         comm_global_isend(destAddr, xferFields*dz, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
         ++emsg ;
      }

      if (rowMin && planeMin) {

            //if(myRank == 0) //printf("8\n");


         int toRank = myRank - domain.tp()*domain.tp() - domain.tp() ;
         destAddr = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dx; ++i) {
               destAddr[i] = (domain.*src)(i) ;
            }
            destAddr += dx ;
         }
         destAddr -= xferFields*dx ;
         comm_global_isend(destAddr, xferFields*dx, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
         ++emsg ;
      }

      if (colMin && planeMin) {

            //if(myRank == 0) //printf("9\n");

         int toRank = myRank - domain.tp()*domain.tp() - 1 ;
         destAddr = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dy; ++i) {
               destAddr[i] = (domain.*src)(i*dx) ;
            }
            destAddr += dy ;
         }
         destAddr -= xferFields*dy ;
         comm_global_isend(destAddr, xferFields*dy, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
         ++emsg ;
      }

      if (rowMax && colMax && doSend) {
            //if(myRank == 0) //printf("10\n");


         int toRank = myRank + domain.tp() + 1 ;
         destAddr = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               destAddr[i] = (domain.*src)(dx*dy - 1 + i*dx*dy) ;
            }
            destAddr += dz ;
         }
         destAddr -= xferFields*dz ;
         comm_global_isend(destAddr, xferFields*dz, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
         ++emsg ;
      }

      if (rowMax && planeMax && doSend) {

            //if(myRank == 0) //printf("11\n");


         int toRank = myRank + domain.tp()*domain.tp() + domain.tp() ;
         destAddr = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dx; ++i) {
              destAddr[i] = (domain.*src)(dx*(dy-1) + dx*dy*(dz-1) + i) ;
            }
            destAddr += dx ;
         }
         destAddr -= xferFields*dx ;
         comm_global_isend(destAddr, xferFields*dx, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
         ++emsg ;
      }

      if (colMax && planeMax && doSend) {
            //if(myRank == 0) //printf("12\n");


         int toRank = myRank + domain.tp()*domain.tp() + 1 ;
         destAddr = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dy; ++i) {
               destAddr[i] = (domain.*src)(dx*dy*(dz-1) + dx - 1 + i*dx) ;
            }
            destAddr += dy ;
         }
         destAddr -= xferFields*dy ;
         comm_global_isend(destAddr, xferFields*dy, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
         ++emsg ;
      }

      if (rowMax && colMin && doSend) {
            //if(myRank == 0) //printf("13\n");


         int toRank = myRank + domain.tp() - 1 ;
         destAddr = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               destAddr[i] = (domain.*src)(dx*(dy-1) + i*dx*dy) ;
            }
            destAddr += dz ;
         }
         destAddr -= xferFields*dz ;
         comm_global_isend(destAddr, xferFields*dz, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
         ++emsg ;
      }

      if (rowMin && planeMax && doSend) {
            //if(myRank == 0) //printf("14\n");


         int toRank = myRank + domain.tp()*domain.tp() - domain.tp() ;
         destAddr = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dx; ++i) {
               destAddr[i] = (domain.*src)(dx*dy*(dz-1) + i) ;
            }
            destAddr += dx ;
         }
         destAddr -= xferFields*dx ;
         comm_global_isend(destAddr, xferFields*dx, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
         ++emsg ;
      }

      if (colMin && planeMax && doSend) {
            //if(myRank == 0) //printf("15\n");


         int toRank = myRank + domain.tp()*domain.tp() - 1 ;
         destAddr = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dy; ++i) {
               destAddr[i] = (domain.*src)(dx*dy*(dz-1) + i*dx) ;
            }
            destAddr += dy ;
         }
         destAddr -= xferFields*dy ;
         comm_global_isend(destAddr, xferFields*dy, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
         ++emsg ;
      }

      if (rowMin && colMax) {

            //if(myRank == 0) //printf("16\n");


         int toRank = myRank - domain.tp() + 1 ;
         destAddr = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               destAddr[i] = (domain.*src)(dx - 1 + i*dx*dy) ;
            }
            destAddr += dz ;
         }
         destAddr -= xferFields*dz;
         comm_global_isend(destAddr, xferFields*dz, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
         ++emsg ;
      }

      if (rowMax && planeMin) {

            //if(myRank == 0) //printf("17\n");


         int toRank = myRank - domain.tp()*domain.tp() + domain.tp() ;
         destAddr = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dx; ++i) {
               destAddr[i] = (domain.*src)(dx*(dy - 1) + i) ;
            }
            destAddr += dx ;
         }
         destAddr -= xferFields*dx ;
         comm_global_isend(destAddr, xferFields*dx, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
         ++emsg ;
      }

      if (colMax && planeMin) {

            //if(myRank == 0) //printf("18\n");

         int toRank = myRank - domain.tp()*domain.tp() + 1 ;
         destAddr = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dy; ++i) {
               destAddr[i] = (domain.*src)(dx - 1 + i*dx) ;
            }
            destAddr += dy ;
         }
         destAddr -= xferFields*dy;
         comm_global_isend(destAddr, xferFields*dy, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg], pmsg+emsg+(26*typeBuf)) ;
         /*
         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         */
         ++emsg ;
      }

      if (rowMin && colMin && planeMin) {
            //if(myRank == 0) //printf("19\n");


         /* corner at domain logical coord (0, 0, 0) */
         int toRank = myRank - domain.tp()*domain.tp() - domain.tp() - 1 ;
         Real_t *comBuf = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = (domain.*fieldData[fi])(0) ;
         }
         comm_global_isend(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf)) ;
         /*
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         */
         ++cmsg ;
      }
      if (rowMin && colMin && planeMax && doSend) {

            //if(myRank == 0) //printf("20\n");


         /* corner at domain logical coord (0, 0, 1) */
         int toRank = myRank + domain.tp()*domain.tp() - domain.tp() - 1 ;
         Real_t *comBuf = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
         Index_t idx = dx*dy*(dz - 1) ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = (domain.*fieldData[fi])(idx) ;
         }
         comm_global_isend(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf)) ;
         /*
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         */
         ++cmsg ;
      }
      if (rowMin && colMax && planeMin) {

        //if(myRank == 0) //printf("21\n");

         /* corner at domain logical coord (1, 0, 0) */
         int toRank = myRank - domain.tp()*domain.tp() - domain.tp() + 1 ;
         Real_t *comBuf = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
         Index_t idx = dx - 1 ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = (domain.*fieldData[fi])(idx) ;
         }
         comm_global_isend(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf)) ;
         /*
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         */
         ++cmsg ;
      }
      if (rowMin && colMax && planeMax && doSend) {

        //if(myRank == 0) //printf("21\n");

         /* corner at domain logical coord (1, 0, 1) */
         int toRank = myRank + domain.tp()*domain.tp() - domain.tp() + 1 ;
         Real_t *comBuf = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
         Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = (domain.*fieldData[fi])(idx) ;
         }
         comm_global_isend(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf)) ;
         /*
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         */
         ++cmsg ;
      }
      if (rowMax && colMin && planeMin) {

        //if(myRank == 0) //printf("22\n");

         /* corner at domain logical coord (0, 1, 0) */
         int toRank = myRank - domain.tp()*domain.tp() + domain.tp() - 1 ;
         Real_t *comBuf = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
         Index_t idx = dx*(dy - 1) ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = (domain.*fieldData[fi])(idx) ;
         }
         comm_global_isend(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf)) ;
         /*
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         */
         ++cmsg ;
      }
      if (rowMax && colMin && planeMax && doSend) {

        //if(myRank == 0) //printf("23\n");
         /* corner at domain logical coord (0, 1, 1) */
         int toRank = myRank + domain.tp()*domain.tp() + domain.tp() - 1 ;
         Real_t *comBuf = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
         Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = (domain.*fieldData[fi])(idx) ;
         }
         comm_global_isend(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf)) ;
         /*
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         */
         ++cmsg ;
      }
      if (rowMax && colMax && planeMin) {

        //if(myRank == 0) //printf("24\n");

         /* corner at domain logical coord (1, 1, 0) */
         int toRank = myRank - domain.tp()*domain.tp() + domain.tp() + 1 ;
         Real_t *comBuf = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
         Index_t idx = dx*dy - 1 ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = (domain.*fieldData[fi])(idx) ;
         }
         comm_global_isend(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf)) ;
         /*
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         */
         ++cmsg ;
      }
      if (rowMax && colMax && planeMax && doSend) {

         /* corner at domain logical coord (1, 1, 1) */
         int toRank = myRank + domain.tp()*domain.tp() + domain.tp() + 1 ;

         //if(myRank == 0)  //printf("25: pmsg: %d, emsg: %d, cmsg: %d\n", pmsg, emsg, cmsg);
         Real_t *comBuf = domain.commDataSend_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
         Index_t idx = dx*dy*dz - 1 ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            comBuf[fi] = (domain.*fieldData[fi])(idx) ;
         }
         comm_global_isend(comBuf, xferFields, baseType,
                   toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg], pmsg+emsg+cmsg+(26*typeBuf)) ;
         /*
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         */
         ++cmsg ;
      }
   }

   //if(myRank == 0)
    //printf("\n***CommSend END: %d sends***\n", pmsg+emsg+cmsg);
  
    comm_global_wait_all(domain.sendRequest, status, pmsg+emsg+cmsg, SEND_REQUEST);

POP_RANGE;

//   MPI_Waitall(26, domain.sendRequest, status) ;
}

/******************************************/

void CommSBN(Domain& domain, int xferFields, Domain_member *fieldData, int typeBuf) {

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

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

//if(myRank == 0)
    //printf("\n *** CommSBN start ***\n");

   PUSH_RANGE("CommSBN", 5);

   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dy ;

      if (planeMin) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
         comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(i) += srcAddr[i] ;
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (planeMax) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
         comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(dx*dy*(dz - 1) + i) += srcAddr[i] ;
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dz ;

      if (rowMin) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
         comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dx; ++j) {
                  (domain.*dest)(i*dx*dy + j) += srcAddr[i*dx + j] ;
               }
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (rowMax) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
         comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dx; ++j) {
                  (domain.*dest)(dx*(dy - 1) + i*dx*dy + j) += srcAddr[i*dx + j] ;
               }
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dy * dz ;

      if (colMin) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
         comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  (domain.*dest)(i*dx*dy + j*dx) += srcAddr[i*dy + j] ;
               }
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (colMax) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
         comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  (domain.*dest)(dx - 1 + i*dx*dy + j*dx) += srcAddr[i*dy + j] ;
               }
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin & colMin) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            (domain.*dest)(i*dx*dy) += srcAddr[i] ;
         }
         srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMin & planeMin) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            (domain.*dest)(i) += srcAddr[i] ;
         }
         srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMin & planeMin) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            (domain.*dest)(i*dx) += srcAddr[i] ;
         }
         srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMax & colMax) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            (domain.*dest)(dx*dy - 1 + i*dx*dy) += srcAddr[i] ;
         }
         srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMax & planeMax) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            (domain.*dest)(dx*(dy-1) + dx*dy*(dz-1) + i) += srcAddr[i] ;
         }
         srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMax & planeMax) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            (domain.*dest)(dx*dy*(dz-1) + dx - 1 + i*dx) += srcAddr[i] ;
         }
         srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMax & colMin) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            (domain.*dest)(dx*(dy-1) + i*dx*dy) += srcAddr[i] ;
         }
         srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMin & planeMax) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            (domain.*dest)(dx*dy*(dz-1) + i) += srcAddr[i] ;
         }
         srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMin & planeMax) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            (domain.*dest)(dx*dy*(dz-1) + i*dx) += srcAddr[i] ;
         }
         srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMin & colMax) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            (domain.*dest)(dx - 1 + i*dx*dy) += srcAddr[i] ;
         }
         srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMax & planeMin) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            (domain.*dest)(dx*(dy - 1) + i) += srcAddr[i] ;
         }
         srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMax & planeMin) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            (domain.*dest)(dx - 1 + i*dx) += srcAddr[i] ;
         }
         srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMin & colMin & planeMin) {
      /* corner at domain logical coord (0, 0, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(0) += comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMin & colMin & planeMax) {
      /* corner at domain logical coord (0, 0, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*(dz - 1) ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) += comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMin & colMax & planeMin) {
      /* corner at domain logical coord (1, 0, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx - 1 ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) += comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMin & colMax & planeMax) {
      /* corner at domain logical coord (1, 0, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) += comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMax & colMin & planeMin) {
      /* corner at domain logical coord (0, 1, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*(dy - 1) ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) += comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMax & colMin & planeMax) {
      /* corner at domain logical coord (0, 1, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) += comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMax & colMax & planeMin) {
      /* corner at domain logical coord (1, 1, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy - 1 ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) += comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMax & colMax & planeMax) {
      /* corner at domain logical coord (1, 1, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*dz - 1 ;
      comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) += comBuf[fi] ;
      }
      ++cmsg ;
   }

   POP_RANGE;
}

/******************************************/

void CommSyncPosVel(Domain& domain, int typeBuf) {

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

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

   PUSH_RANGE("CommSyncPosVel", 6);

//if(myRank == 0)
    //printf("\n *** CommSyncPosVel start ***\n");

   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dy ;

      if (planeMin && doRecv) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
               comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(i) = srcAddr[i] ;
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (planeMax) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
               comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(dx*dy*(dz - 1) + i) = srcAddr[i] ;
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dz ;

      if (rowMin && doRecv) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
               comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dx; ++j) {
                  (domain.*dest)(i*dx*dy + j) = srcAddr[i*dx + j] ;
               }
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (rowMax) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
               comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dx; ++j) {
                  (domain.*dest)(dx*(dy - 1) + i*dx*dy + j) = srcAddr[i*dx + j] ;
               }
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dy * dz ;

      if (colMin && doRecv) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
               comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  (domain.*dest)(i*dx*dy + j*dx) = srcAddr[i*dy + j] ;
               }
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (colMax) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
               comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  (domain.*dest)(dx - 1 + i*dx*dy + j*dx) = srcAddr[i*dy + j] ;
               }
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin && colMin && doRecv) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            (domain.*dest)(i*dx*dy) = srcAddr[i] ;
         }
         srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMin && planeMin && doRecv) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            (domain.*dest)(i) = srcAddr[i] ;
         }
         srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMin && planeMin && doRecv) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            (domain.*dest)(i*dx) = srcAddr[i] ;
         }
         srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMax && colMax) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            (domain.*dest)(dx*dy - 1 + i*dx*dy) = srcAddr[i] ;
         }
         srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMax && planeMax) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            (domain.*dest)(dx*(dy-1) + dx*dy*(dz-1) + i) = srcAddr[i] ;
         }
         srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMax && planeMax) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            (domain.*dest)(dx*dy*(dz-1) + dx - 1 + i*dx) = srcAddr[i] ;
         }
         srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMax && colMin) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            (domain.*dest)(dx*(dy-1) + i*dx*dy) = srcAddr[i] ;
         }
         srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMin && planeMax) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            (domain.*dest)(dx*dy*(dz-1) + i) = srcAddr[i] ;
         }
         srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMin && planeMax) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            (domain.*dest)(dx*dy*(dz-1) + i*dx) = srcAddr[i] ;
         }
         srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMin && colMax && doRecv) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            (domain.*dest)(dx - 1 + i*dx*dy) = srcAddr[i] ;
         }
         srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMax && planeMin && doRecv) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            (domain.*dest)(dx*(dy - 1) + i) = srcAddr[i] ;
         }
         srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMax && planeMin && doRecv) {
      srcAddr = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg] ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg], &status, RECV_REQUEST, pmsg+emsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            (domain.*dest)(dx - 1 + i*dx) = srcAddr[i] ;
         }
         srcAddr += dy ;
      }
      ++emsg ;
   }


   if (rowMin && colMin && planeMin && doRecv) {
      /* corner at domain logical coord (0, 0, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(0) = comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMin && colMin && planeMax) {
      /* corner at domain logical coord (0, 0, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*(dz - 1) ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) = comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMin && colMax && planeMin && doRecv) {
      /* corner at domain logical coord (1, 0, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx - 1 ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) = comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMin && colMax && planeMax) {
      /* corner at domain logical coord (1, 0, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) = comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMax && colMin && planeMin && doRecv) {
      /* corner at domain logical coord (0, 1, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*(dy - 1) ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) = comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMax && colMin && planeMax) {
      /* corner at domain logical coord (0, 1, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) = comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMax && colMax && planeMin && doRecv) {
      /* corner at domain logical coord (1, 1, 0) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy - 1 ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) = comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMax && colMax && planeMax) {
      /* corner at domain logical coord (1, 1, 1) */
      Real_t *comBuf = domain.commDataRecv_multi[(26*typeBuf)+pmsg + emsg + cmsg] ;
      Index_t idx = dx*dy*dz - 1 ;
            comm_global_wait(&domain.recvRequest[pmsg+emsg+cmsg], &status, RECV_REQUEST, pmsg+emsg+cmsg+(26*typeBuf)) ;
      //MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) = comBuf[fi] ;
      }
      ++cmsg ;
   }

   POP_RANGE;

}

/******************************************/

void CommMonoQ(Domain& domain, int typeBuf)
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

//if(myRank == 0)
    //printf("\n *** CommMonoQ start ***\n");

PUSH_RANGE("CommMonoQ", 7);

   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dy ;

      if (planeMin) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
            }
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
      if (planeMax) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
            }
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
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
         comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
            }
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
      if (rowMax) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
         comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
            }
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
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
         comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
            }
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
      if (colMax) {
         /* contiguous memory */
         srcAddr = domain.commDataRecv_multi[pmsg+(26*typeBuf)] ;
         comm_global_wait(&domain.recvRequest[pmsg], &status, RECV_REQUEST, pmsg+(26*typeBuf)) ;
         //MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }
   POP_RANGE;

}

#endif
