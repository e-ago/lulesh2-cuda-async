#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <errno.h>
//#include <mpi.h>
#include <cuda_runtime.h>
#include "mp.h"
#include "comm.h"
//#include "debug.h"

#define DBG(FMT, ARGS...) /*       \
    do {                                                                \
        if (comm_rank == 0) {                                            \
            fprintf(stdout, "[%d] [%d] LULESH %s(): " FMT,               \
                    getpid(), comm_rank, __FUNCTION__ , ## ARGS);   \
            fflush(stdout);                                             \
        }                                                               \
    } while(0)
*/
#define MP_CHECK(stmt)                                  \
do {                                                    \
    int result = (stmt);                                \
    if (0 != result) {                                  \
        fprintf(stderr, "[%s:%d] mp call failed \n",    \
         __FILE__, __LINE__);                           \
        exit(-1);                                       \
    }                                                   \
    assert(0 == result);                                \
} while (0)


#define comm_err(FMT, ARGS...)  do { fprintf(stderr, "ERR [%d] %s() " FMT, comm_rank, __FUNCTION__ , ## ARGS); fflush(stderr); } while(0)


#ifndef MAX
#define MAX(A,B) ((A)>(B)?(A):(B))
#endif

#define MAX_RANKS 128

static int         comm_initialized = 0;
static int         rank_to_peer[MAX_RANKS] = {0,};
static int         peers[MAX_RANKS] = {0,};
static int         n_peers = -1;
static const int   bad_peer = -1;
static int         comm_size;
static int         comm_rank;

// tables are indexed by rank, not peer
static uint32_t   *ready_table;
static mp_reg_t    ready_table_reg;
static mp_window_t ready_table_win;

static uint32_t   *ready_values;
static uint32_t   *remote_ready_values;
static mp_reg_t    remote_ready_values_reg;

static mp_request_t reqs[MAX_REQS];
static int n_reqs = 0;

#define PAGE_BITS 12
#define PAGE_SIZE (1ULL<<PAGE_BITS)
#define PAGE_OFF  (PAGE_SIZE-1)
#define PAGE_MASK (~(PAGE_OFF))

#define mb()    __asm__ volatile("mfence":::"memory")
#define rmb()   __asm__ volatile("lfence":::"memory")
#define wmb()   __asm__ volatile("sfence" ::: "memory")
#define iomb() mb()

#define ACCESS_ONCE(V)                          \
    (*(volatile __typeof__ (V) *)&(V))

static inline void arch_pause(void)
{
        __asm__ volatile("pause\n": : :"memory");
}


static int curr_recv_request=0;
static int curr_send_request=0;
static int curr_send_stream_request=0;
static int first_recv_request=0;
static int first_send_request=0;
static int first_send_stream_request=0;
static int first_timer_recv_request=0;

static int max_send_request=0;
static int max_send_stream_request=0;
static int max_recv_request=0;

/*
static comm_request_t  * recv_requests;
static comm_request_t  * send_requests;
static comm_request_t  * send_stream_requests;
static comm_request_t  * ready_requests;
*/

static comm_request_t recv_requests[MAX_REQS];
static comm_request_t send_requests[MAX_REQS];
static comm_request_t send_stream_requests[MAX_REQS];
static comm_request_t ready_requests[MAX_REQS];

static comm_request_t timer_recv_requests[MAX_REQS];
static comm_request_t timer_send_requests[MAX_REQS];

static comm_reg_t * recv_region[80]; //26*3
static comm_reg_t * send_region[80];
static comm_reg_t * send_stream_region[80];

static comm_reg_t * timer_recv_region[1];
static comm_reg_t * timer_send_region[1];

static int print=1;

static int indexRecv=0;
static int indexSend=0;
static int indexSendStream=0;
static int indexTimerRecv=0;
static int indexTimerSend=0;

static int indexGpuReqs=0;

static int maxBufSize=0;
static int globalIndexSendRegion=0;
static int globalIndexRecvRegion=0;
static int indexGlobalReady=0;

static int startGlobalReqsIndex=0;

static comm_dev_descs_t pdreqs = NULL;
static const size_t n_dreqs = 512; //36 prima
static int dreq_idx = 0;

static inline void arch_cpu_relax(void)
{
        //rmb();
        //arch_rep_nop();
        arch_pause();
        //arch_pause(); arch_pause();
        //BUG: poll_lat hangs after a few iterations
        //arch_wait();
}

int comm_use_comm()
{
    static int use_comm = -1;
    if (-1 == use_comm) {
        const char *env = getenv("COMM_USE_COMM");
        if (env) {
            use_comm = !!atoi(env);
            printf("WARNING: %s Comm-based communications\n", (use_comm)?"enabling":"disabling");
        } else
            use_comm = 0; // default
    }
    return use_comm;
}

int comm_use_gdrdma()
{
    static int use_gdrdma = -1;
    if (-1 == use_gdrdma) {
        const char *env = getenv("COMM_USE_GDRDMA");
        if (env) {
            use_gdrdma = !!atoi(env);
            printf("WARNING: %s CUDA-aware/RDMA communications\n", (use_gdrdma)?"enabling":"disabling");
        } else
            use_gdrdma = 0; // default
    }
    return use_gdrdma;
}

int comm_use_gpu_comm()
{
    static int use_gpu_comm = -1;
    if (-1 == use_gpu_comm) {
        const char *env = getenv("COMM_USE_GPU_COMM");
        if (env) {
            use_gpu_comm = !!atoi(env);
            printf("WARNING: %s GPU-initiated communications\n", (use_gpu_comm)?"enabling":"disabling");
        } else
            use_gpu_comm = 0; // default
    }
    return use_gpu_comm;
}

int comm_use_async()
{
    static int use_async = -1;
    if (-1 == use_async) {
        const char *env = getenv("COMM_USE_ASYNC");
        if (env) {
            use_async = !!atoi(env);
            printf("WARNING: %s GPUDirect Async for communications\n", (use_async)?"enabling":"disabling");
        } else
            use_async = 0; // default
    }
    return use_async;
}

int comm_init(MPI_Comm comm)
{
    int i, j;

    MPI_Comm_size (comm, &comm_size);
    MPI_Comm_rank (comm, &comm_rank);

    assert(comm_size < MAX_RANKS);

    // init peers    
    for (i=0, j=0; i<comm_size; ++i) {
        if (i!=comm_rank) {
            peers[j] = i;
            rank_to_peer[i] = j;
            DBG("peers[%d]=rank %d\n", j, i);
            ++j;
        } else {
            // self reference is forbidden
            rank_to_peer[i] = bad_peer;
        }
    }
    n_peers = j;
    assert(comm_size-1 == n_peers);
    DBG("n_peers=%d\n", n_peers);

    MP_CHECK(mp_init(comm, peers, n_peers));
#if 0
    // init ready stuff
    size_t table_size = MAX(sizeof(*ready_table) * comm_size, PAGE_SIZE);
    ready_table = (uint32_t *)memalign(PAGE_SIZE, table_size);
    assert(ready_table);

    ready_values = (uint32_t *)malloc(table_size);
    assert(ready_values);

    remote_ready_values = (uint32_t *)memalign(PAGE_SIZE, table_size);
    assert(remote_ready_values);
    
    for (i=0; i<comm_size; ++i) {
        ready_table[i] = 0;  // remotely written table
        ready_values[i] = 1; // locally expected value
        remote_ready_values[i] = 1; // value to be sent remotely
    }
    iomb();
#endif

#if 0
    if(comm_rank == 6)
    printf("registering ready_table size=%d\n", table_size);
    MP_CHECK(mp_register(ready_table, table_size, &ready_table_reg));
    if(comm_rank == 6)
    printf("creating ready_table window\n");
    MP_CHECK(mp_window_create(ready_table, table_size, &ready_table_win));
    if(comm_rank == 6)
    printf("registering remote_ready_table\n");
    MP_CHECK(mp_register(remote_ready_values, table_size, &remote_ready_values_reg));
#endif

    comm_initialized = 1;

    return 0;
}

void comm_finalize() {
    mp_finalize();
}

static size_t comm_size_of_mpi_type(MPI_Datatype mpi_type)
{
    size_t ret = 0;

    if (mpi_type == MPI_DOUBLE) {
        ret = sizeof(double);
    } 
    else if (mpi_type == MPI_FLOAT) {
        ret = sizeof(float);
    }
    else if (mpi_type == MPI_CHAR) {
        ret = sizeof(char);
    }
    else if (mpi_type == MPI_BYTE) {
        ret = sizeof(char);
    }
    else if (mpi_type == MPI_INT) {
        ret = sizeof(int);
    }
    else {
        ret = sizeof(mpi_type);
        comm_err("invalid type\n");
        exit(1);
    }
    return ret;
}

static int comm_mpi_rank_to_peer(int rank)
{
    assert(comm_initialized);
    assert(rank < comm_size);
    assert(rank >= 0);
    int peer = rank;
    //int peer = rank_to_peer[rank];
    //assert(peer != bad_peer);
    //assert(peer < n_peers);
    return peer;
}

static mp_reg_t *comm_reg(void *ptr, size_t size)
{
    assert(comm_initialized);
    return NULL;
}

static void atomic_inc(uint32_t *ptr)
{
    __sync_fetch_and_add(ptr, 1);
    //++ACCESS_ONCE(*ptr);
    iomb();
}

static void comm_track_request(mp_request_t *req)
{

    //assert(n_reqs < MAX_REQS);    
    reqs[n_reqs++] = *req;
//    n_reqs = (n_reqs+1)%MAX_REQS;

    DBG("n_reqs=%d\n", n_reqs);
}

int comm_send_ready_on_stream(int rank, comm_request_t *creq, cudaStream_t stream)
{
    assert(comm_initialized);
    assert(rank < comm_size);
    int ret = 0;
    int peer = comm_mpi_rank_to_peer(rank);
    mp_request_t *req = (mp_request_t*)creq;
    assert(req);
    int remote_offset = /*self rank*/comm_rank * sizeof(uint32_t);
    DBG("dest_rank=%d payload=%x offset=%d\n", rank, remote_ready_values[rank], remote_offset);
    MP_CHECK(mp_iput_on_stream(&remote_ready_values[rank], sizeof(uint32_t), &remote_ready_values_reg, 
                               peer, remote_offset, &ready_table_win, req, MP_PUT_INLINE, stream));
    //MP_CHECK(mp_wait(req));
    comm_track_request(req);
    atomic_inc(&remote_ready_values[rank]);
}

int comm_send_ready(int rank, comm_request_t *creq)
{
    assert(comm_initialized);
    assert(rank < comm_size);
    int ret = 0;
    int peer = comm_mpi_rank_to_peer(rank);
    mp_request_t *req = (mp_request_t*)creq;
    assert(req);
    int remote_offset = /*my rank*/comm_rank * sizeof(uint32_t);
    DBG("dest_rank=%d payload=%x offset=%d\n", rank, remote_ready_values[rank], remote_offset);
    MP_CHECK(mp_iput(&remote_ready_values[rank], sizeof(uint32_t), &remote_ready_values_reg, 
                     peer, remote_offset, &ready_table_win, req, MP_PUT_INLINE));
    //MP_CHECK(mp_wait(req));
    comm_track_request(req);
    atomic_inc(&remote_ready_values[rank]);
}

//ELENAGO DA QUI
int comm_send_ready_setup(int rank)
{
    comm_send_ready(rank, &ready_requests[indexGlobalReady]);
    indexGlobalReady++;
}

int comm_send_ready_stream_setup(int rank, cudaStream_t stream)
{
    comm_send_ready_on_stream(rank, &ready_requests[indexGlobalReady], stream);
    indexGlobalReady++;
}

#if 0
int comm_put_value(int rank, double *valueToSend, comm_request_t *creq) 
{
    assert(comm_initialized);
    assert(rank < comm_size);
    int ret = 0;
    int peer = comm_mpi_rank_to_peer(rank);
    mp_request_t *req = (mp_request_t*)creq;
    assert(req);
    int remote_offset = comm_rank * sizeof(uint64_t); /*my rank*/
    if(comm_rank == 6)
        printf("dest_rank=%d valueToSend=%12.6e offset=%d\n", 
            peer, valueToSend[0], remote_offset);

/* remote_ready_values[rank]*/

    DBG("dest_rank=%d payload=%x offset=%d\n", rank, remote_ready_values[rank], remote_offset);
    MP_CHECK(mp_iput(valueToSend, sizeof(uint64_t), &remote_ready_values_reg, 
                     peer, remote_offset, &ready_table_win, req, MP_PUT_INLINE));
    //MP_CHECK(mp_wait(req));
    comm_track_request(req);
    //atomic_inc(&remote_ready_values[rank]);
}
#endif

int comm_wait_ready_on_stream(int rank, cudaStream_t stream)
{
    assert(comm_initialized);
    assert(rank < comm_size);
    int ret = 0;
    int peer = comm_mpi_rank_to_peer(rank);
    DBG("rank=%d payload=%x\n", rank, ready_values[rank]); 
    MP_CHECK(mp_wait_dword_geq_on_stream(&ready_table[rank], ready_values[rank], stream));
    ready_values[rank]++;
    return ret;
}

int comm_wait_ready(int rank)
{
    assert(comm_initialized);
    assert(rank < comm_size);
    int ret = 0;
    int peer = comm_mpi_rank_to_peer(rank);
    int cnt = 0;
    DBG("rank=%d payload=%x\n", rank, ready_values[rank]);
    while (ACCESS_ONCE(ready_table[rank]) < ready_values[rank]) {
        rmb();
        arch_cpu_relax();
        ++cnt;
        if (cnt > 10000) {
            comm_progress();
            cnt = 0;
        }
    }
    ready_values[rank]++;
    return ret;
}

#if 0
double comm_wait_put_value(int rank, double myValue)
{
    double value = myValue;
    assert(comm_initialized);
    assert(rank < comm_size);
    int ret = 0;
    int peer = comm_mpi_rank_to_peer(rank);
    int cnt = 0;
    if(comm_rank == 6)
        printf("Waiting for rank %d\n", rank);

    DBG("rank=%d payload=%x\n", rank, ready_values[rank]);
    while (ACCESS_ONCE(ready_table[rank]) == -1) { //< ready_values[rank]) {
        rmb();
        arch_cpu_relax();
        ++cnt;
        if (cnt > 10000) {
            comm_progress();
            cnt = 0;
        }
    }

    if(comm_rank == 6)
        printf("myValue: %12.6e value: %12.6e ready_table[%d]: %12.6e\n",
         myValue, value, rank, ready_table[rank]);

    if(ready_table[rank] < value)
        value = ready_table[rank];

    return value;
    //ready_values[rank]++;
}

double comm_min_op_put_value(int numRank, int myRank, double myValue)
{
    assert(comm_initialized);
    int i;
    double value=myValue;
    for(i=0; i<numRank; i++)
    {
        if(comm_rank == 6)
           printf("ready_table[%d]: %12.6e, value: %12.6e\n", i, ready_table[i], value);

        if(i != myRank && ready_table[i] < value)
            value = ready_table[i];
    }

    return value;
}

void comm_cleanup_put_table()
{
    int i;
    assert(comm_initialized);
    assert(ready_table);

    for(i=0; i<comm_size; i++)
        ready_table[i] = -1;
}
#endif
int comm_test_ready(int rank, int *p_rdy)
{
    assert(comm_initialized);
    assert(rank < comm_size);
    int ret = 0;
    int peer = comm_mpi_rank_to_peer(rank);
    static int cnt = 0;
    DBG("rank=%d payload=%x\n", rank, ready_values[rank]);
    do {
        rmb();
        *p_rdy = !(ACCESS_ONCE(ready_table[rank]) < ready_values[rank]);
        if (*p_rdy) {
            ++ready_values[rank];
            break;
        }
        ++cnt;
        if (cnt > 10000) {
            arch_cpu_relax();
            cnt = 0;
        }
    } while(0);
    return ret;
}

int comm_wait_stream_setup(int type, cudaStream_t stream, int index)
{   
#if 0
    if(comm_rank == 6 && print == 1)
            printf("COMM WAIT SEND STREAM index %d first_send_stream_request: %d\n", index, first_send_stream_request);
    
#endif
    if(type == RECV_REQUEST)
    {
//        if(comm_rank == 0)
//            printf("WAIT RECV  on stream, first_recv_request: %d, curr_recv_request: %d\n", first_recv_request, curr_recv_request);
        comm_wait_all_on_stream(1, &recv_requests[first_recv_request], stream);
        curr_recv_request--; //??
        first_recv_request++;
    }
    else if(type == SEND_STREAM_REQUEST)
    {   
//        if(comm_rank == 0)
//            printf("WAIT SEND_STREAM_REQUEST  on stream, first_send_stream_request: %d, curr_send_stream_request: %d\n", first_send_stream_request, curr_send_stream_request);

        comm_wait_all_on_stream(1, &send_stream_requests[first_send_stream_request], stream);
        curr_send_stream_request--; //??
        first_send_stream_request++;
    }
    else
        return -1;

    return 1;
}


int comm_wait_all_stream_setup(int count, int type, cudaStream_t stream)
{   
#if 0
    if(comm_rank == 6 && print == 1)
        printf("COMM WAIT STREAM ALL type: %d Num Recv %d, Num Send: %d\n",  type, curr_recv_request, curr_send_request);
#endif
    if(type == RECV_REQUEST)
    {
        comm_wait_all_on_stream(curr_recv_request, recv_requests+first_recv_request, stream);
        first_recv_request += count;
        curr_recv_request=0; //??
    }
    else if(type == SEND_STREAM_REQUEST)
    {
        comm_wait_all_on_stream(curr_send_stream_request, send_stream_requests+first_recv_request, stream);
        first_send_stream_request += count;
        curr_send_stream_request=0; //??
    }
    else
        return -1;

    return 1;
}

int comm_wait_all_on_stream(int count, comm_request_t *creqs, cudaStream_t stream)
{
    int ret = 0;
    DBG("count=%d\n", count);
    assert(comm_initialized);
    mp_request_t *reqs = (mp_request_t*)creqs;
    if (1 == count) {
        assert(*reqs);
        MP_CHECK(mp_wait_on_stream(reqs, stream));
    } else {
        MP_CHECK(mp_wait_all_on_stream(count, reqs, stream));
    }
   // memset(creqs, 0, sizeof(comm_request_t)*count);
    return ret;
}

int comm_wait_all_setup(int count, int type)
{   
    #if 0

    if(comm_rank == 6)
        printf("COMM WAIT ALL type: %d Num Recv %d, First Recv: %d, Num Send: %d, First Send: %d\n", 
            type,
            curr_recv_request, first_recv_request, 
            curr_send_request, first_send_request);
    #endif

    if(type == RECV_REQUEST)
    {
        comm_wait_all(count, recv_requests+first_recv_request);
        first_recv_request += curr_recv_request;
        curr_recv_request=0; //??
    }
    else if(type == SEND_REQUEST)
    {
        comm_wait_all(count, send_requests+first_send_request);
        first_send_request += curr_send_request;
        curr_send_request=0; //??
    }
    else 
        return -1;

    return 1;
}

int comm_wait_all(int count, comm_request_t *creqs)
{
    int ret = 0;
    DBG("count=%d\n", count);
    assert(comm_initialized);
    mp_request_t *reqs = (mp_request_t*)creqs;
    MP_CHECK(mp_wait_all(count, reqs));
    memset(creqs, 0, sizeof(comm_request_t)*count);
    return ret;
}

int comm_wait_setup(int type, int index)
{   
    if(type == RECV_REQUEST)
    {   
        #if 0
        if(comm_rank == 6 && print == 1)
            printf("COMM WAIT RECV REQ index %d first_recv_request %d \n", index, first_recv_request);
        #endif
        comm_wait(&recv_requests[first_recv_request]);
        curr_recv_request--; //??
        first_recv_request++;
    }
    else if(type == SEND_REQUEST)
    {
        #if 0
        if(comm_rank == 6 && print == 1)
            printf("COMM WAIT SEND REQ index %d first_send_request: %d\n", index, first_send_request);
        #endif
        comm_wait(&send_requests[first_send_request]);
        curr_send_request--; //??
        first_send_request++;
    }
    else 
        return -1;

    return 1;
}

int comm_wait(comm_request_t *creq)
{
    int ret = 0;
    assert(comm_initialized);
    mp_request_t *req = (mp_request_t*)creq;
    MP_CHECK(mp_wait(req));
    memset(creq, 0, sizeof(comm_request_t));
    return ret;
}

int comm_reset(int type)
{
    int i=0;
#if 0
    if(comm_rank == 6 && print == 1)
        printf("***** COMM RESET type %d!!\n", type);
#endif
    if(type == RECV_REQUEST)
    {
        if(max_recv_request > 0)
        {
            //free(recv_requests);
            //free(recv_region);
            /*
            for(i=0; i<max_recv_request; i++)
                mp_deregister(&(recv_region[i]));
            */
            curr_recv_request=0;
            first_recv_request=0;
            max_recv_request=0;
        }
    }
    
    if(type == SEND_REQUEST)
    {
        if(comm_use_async())
        {

#if 0
            if(comm_rank == 6 && print == 1)
                printf("-- RESET ASYNC max_send_stream_request: %d!!\n", max_send_stream_request);
#endif
            if(max_send_stream_request > 0)
            {
                //free(send_stream_requests);
                //free(send_stream_region);

                curr_send_stream_request=0;
                first_send_stream_request=0;
                //max_send_stream_request=0;
            }
        }
        else
        {
#if 0
                    if(comm_rank == 6 && print == 1)
                printf("-- RESET SYNC max_send_request: %d!!\n", max_send_request);
#endif

            if(max_send_request > 0)
            {
                //free(send_requests);
                //free(send_region);

                curr_send_request=0;
                first_send_request=0;
                //max_send_request=0;
            }
        }
    }

    if(type == SEND_STREAM_REQUEST)
    {
        if(max_send_request > 0)
        {
            //free(send_stream_requests);
            //free(send_stream_region);

            curr_send_stream_request=0;
            first_send_stream_request=0;
            //max_send_stream_request=0;
        }
    }

    return 1;
}

int comm_requests_setup(int numReq, int type)
{
    #if 0
    if(comm_rank == 6 && print == 1)
        printf("COMM SETUP %d REQUESTS TYPE %d\n", numReq, type);
    #endif

#if 0
    if(type == RECV_REQUEST)
    {
        recv_requests = (comm_request_t *) calloc(numReq, sizeof(comm_request_t));
        max_recv_request=numReq;
        curr_recv_request=0;
    }

    if(type == SEND_REQUEST)
    {
        send_requests = (comm_request_t *) calloc(numReq, sizeof(comm_request_t));
        max_send_request=numReq;
        curr_send_request=0;
    }

    if(type == SEND_STREAM_REQUEST)
    {
        send_stream_requests = (comm_request_t *) calloc(numReq, sizeof(comm_request_t));
        max_send_stream_request=numReq;
        curr_send_stream_request=0;
    }

    if(type == READY_REQUEST)
        ready_requests = (comm_request_t *) calloc(numReq, sizeof(comm_request_t));
#endif

    indexGpuReqs=0;
    cudaMallocHost((void **)&(pdreqs), n_dreqs*sizeof(comm_dev_descs));

    if(comm_rank == 0)
        printf("pdreqs alloc n_dreqs: %d\n", n_dreqs);

    //cudaHostAlloc(pdreqs, n_dreqs*sizeof(pdreqs), cudaHostAllocPortable | cudaHostAllocMapped );

    return 1;
}

#if 1
int comm_regions_setup(int numReq, int type)
{
#if 0
    if(comm_rank == 0)
        printf("COMM SETUP %d REGIONS TYPE %d\n", numReq, type);
#endif

    if(type == RECV_REGION)
    {
        for(int index=0; index < numReq; index++)
            recv_region[index] = (comm_reg_t*)calloc(1, 1*sizeof(comm_reg_t));
    }
        
    if(type == SEND_REGION)
    {
        for(int index=0; index < numReq; index++)
            send_region[index] = (comm_reg_t*)calloc(1, 1*sizeof(comm_reg_t));
    }
        
    if(type == SEND_STREAM_REGION)
    {
        for(int index=0; index < numReq; index++)
            send_stream_region[index] = (comm_reg_t*)calloc(1, 1*sizeof(comm_reg_t));
    }
    
    if(type == TIMER_RECV_REGION)
    {
        timer_recv_region[0] = (comm_reg_t*)calloc(1, 1*sizeof(comm_reg_t));
    }

    if(type == TIMER_SEND_REGION)
    {
        timer_send_region[0] = (comm_reg_t*)calloc(1, 1*sizeof(comm_reg_t));
    }

    return 1;
}
#endif

#if 0
int comm_regions_setup(int index, int type)
{
#if 0
    if(comm_rank == 6 && print == 1)
        printf("COMM SETUP %d REGIONS TYPE %d\n", numReq, type);
#endif

    if(type == RECV_REGION)
        recv_region[index] = (comm_reg_t*)calloc(1, 1*sizeof(comm_reg_t));

    if(type == SEND_REGION)
        send_region[index] = (comm_reg_t*)calloc(1, 1*sizeof(comm_reg_t));
        
    if(type == SEND_STREAM_REGION)
        send_stream_region[index] = (comm_reg_t*)calloc(1, 1*sizeof(comm_reg_t));
    
    return 1;
}
#endif

int comm_irecv_setup(void *recv_buf, size_t size, MPI_Datatype type, int src_rank, int index)
{
    
#if 0
    if(comm_rank == 6)
        printf("Rank %d, RECV numero %d: recv_buf=%p, region=%p\n\n",
                comm_rank, index, recv_buf, recv_region[index]);
#endif
#if 0
    if(comm_rank == 6 && print == 1)
        printf("COMM SETUP RECV indexRegion: %d indexRecv: %d address buf=%p, size: %d\n", index, indexRecv, recv_buf, size);
#endif

//    if(comm_rank == 0)
//    printf("RECV request %d, curr: %d region: %d from: %d\n", indexRecv, curr_recv_request, index, src_rank);
 

    comm_irecv(recv_buf, size, type, recv_region[index], src_rank, &(recv_requests[indexRecv]));
    curr_recv_request++;
    indexRecv = (indexRecv+1)%MAX_REQS;
}

// tags are not supported!!!
int comm_irecv(void *recv_buf, size_t size, MPI_Datatype type, comm_reg_t *creg,
               int src_rank, comm_request_t *creq)
{
    assert(comm_initialized);
    int ret = 0;
    int retcode;
    size_t nbytes = size*comm_size_of_mpi_type(type);
    mp_reg_t *reg = (mp_reg_t*)creg;
    assert(reg);
    mp_request_t *req = (mp_request_t*)creq;
    assert(req);
    int peer = comm_mpi_rank_to_peer(src_rank);

//if(comm_rank == 0)
//    printf("src_rank=%d peer=%d nbytes=%d buf=%p *reg=%p\n", src_rank, peer, nbytes, recv_buf, *reg);

    if (!size) {
        ret = -EINVAL;
        comm_err("SIZE==0\n");
        goto out;
    }

    if (!*reg) {
 //       if(comm_rank == 0)
 //       printf("irecv registering buffer %p, size: %d reg=%p\n", recv_buf, nbytes, reg);

        DBG("registering buffer %p\n", recv_buf);
        MP_CHECK(mp_register(recv_buf, nbytes, reg));
    }

    retcode = mp_irecv(recv_buf,
                       nbytes,
                       peer,
                       reg,
                       req);
    if (retcode) {
        comm_err("error in mp_irecv ret=%d\n", retcode);
        ret = -1;
        goto out;
    }
    comm_track_request(req);
out:
    return ret;
}

int comm_isend_stream_setup(void *send_buf, size_t size, MPI_Datatype type, int dest_rank, cudaStream_t stream, int index)
{
#if 0
    //if(comm_rank == 6)
        printf("COMM SETUP SEND STREAM MYRANK=%d, toSend=%d, index %d indexSendStream: %d curr_send_stream_request: %d send_buf= %p, send_region=%p send_req=%p\n", 
            comm_rank, dest_rank, index, indexSendStream, curr_send_stream_request, send_buf, send_stream_region[index], &(send_stream_requests[indexSendStream]) );
#endif

//    if(comm_rank == 0)
//        printf("SEND stream request %d, curr: %d region %d \n", indexSendStream, curr_send_stream_request, index);
    comm_isend_on_stream(send_buf, size, type, send_stream_region[index], dest_rank, &(send_stream_requests[indexSendStream]), stream);
    curr_send_stream_request++;
    indexSendStream = (indexSendStream+1)%MAX_REQS;
}

int comm_isend_on_stream(void *send_buf, size_t size, MPI_Datatype type, comm_reg_t *creg,
                         int dest_rank, comm_request_t *creq, cudaStream_t stream)
{
    assert(comm_initialized);
    int ret = 0;
    int retcode;
    size_t nbytes = size*comm_size_of_mpi_type(type);
    mp_reg_t *reg = (mp_reg_t*)creg;
    assert(reg);
    mp_request_t *req = (mp_request_t*)creq;
    int peer = comm_mpi_rank_to_peer(dest_rank);
#if 0
    if(comm_rank == 6)
        printf("buf=%p reg=%p dest_rank=%d peer=%d size=%d nbytes=%d\n", send_buf, reg, dest_rank, peer, size, nbytes);
#endif
    if (!size) {
        ret = -EINVAL;
        comm_err("SIZE==0\n");
        goto out;
    }

    if (!*reg) {
        //printf("SEND STREAM registering buffer %p\n", send_buf);
        MP_CHECK(mp_register(send_buf, nbytes, reg));
    }
    retcode = mp_isend_on_stream(send_buf,
                                 nbytes,
                                 peer,
                                 reg,
                                 req,
                                 stream);
    if (retcode) {
        comm_err("error in mp_isend_on_stream ret=%d\n", retcode);
        ret = -1;
        goto out;
    }
    comm_track_request(req);
out:
    return ret;
}

int comm_isend_setup(void *send_buf, size_t size, MPI_Datatype type, int dest_rank, int index)
{
#if 0
    if(comm_rank == 6)
        printf("Rank %d, SEND numero %d: send_buf=%p, send_region=%p\n\n",
                comm_rank, index, send_buf, send_region[index]);
#endif////
    comm_isend(send_buf, size, type, send_region[index], dest_rank, &(send_requests[indexSend]));
    curr_send_request++;
    indexSend = (indexSend+1)%MAX_REQS;
}

int comm_isend(void *send_buf, size_t size, MPI_Datatype type, comm_reg_t *creg,
               int dest_rank, comm_request_t *creq)
{
    assert(comm_initialized);
    int ret = 0;
    int retcode;
    size_t nbytes = size*comm_size_of_mpi_type(type);
    mp_reg_t *reg = (mp_reg_t*)creg;
    assert(reg);
    mp_request_t *req = (mp_request_t*)creq;
    int peer = comm_mpi_rank_to_peer(dest_rank);

    DBG("dest_rank=%d peer=%d nbytes=%d\n", dest_rank, peer, nbytes);

    if (!size) {
        ret = -EINVAL;
        comm_err("SIZE==0\n");
        goto out;
    }

    if (!*reg) {
#if 0
    if(comm_rank == 6)
        printf("isend registering buffer %p, size: %d reg=%p\n", send_buf, nbytes, reg);
#endif

        DBG("registering buffer %p\n", send_buf);
        MP_CHECK(mp_register(send_buf, nbytes, reg));
    }

    retcode = mp_isend(send_buf,
                       nbytes,
                       peer,
                       reg,
                       req);
    if (retcode) {
        comm_err("error in mp_isend ret=%d\n", retcode);
        ret = -1;
        goto out;
    }
    comm_track_request(req);
out:
    return ret;
}

int comm_register(void *buf, size_t size, int type)
{
    assert(comm_initialized);
    int ret = 0;
    int retcode;
    size_t nbytes = size; //*comm_size_of_mpi_type(MPI_DOUBLE);
    mp_reg_t *reg;

    if(type == RECV_REQUEST)
        reg = (mp_reg_t*)recv_region;
    if(type == SEND_REQUEST)
        reg = (mp_reg_t*)send_region;
    if(type == SEND_STREAM_REQUEST)
        reg = (mp_reg_t*)send_stream_region;

    assert(reg);

    if (!size) {
        ret = -EINVAL;
        comm_err("SIZE==0\n");
        goto out;
    }

    if (!*reg) {
   #if 0
         if(comm_rank == 6)
            printf("registering buffer %p with region %p, type: %d\n", buf, reg, type);
   #endif
        MP_CHECK(mp_register(buf, nbytes, reg));
    }

out:
    return ret;
}

int comm_register_index(void *buf, size_t size, int type, int index)
{
    assert(comm_initialized);
    int ret = 0;
    int retcode;
    size_t nbytes = size; //*comm_size_of_mpi_type(MPI_DOUBLE);
    mp_reg_t *reg;

    if(type == RECV_REGION)
        reg = (mp_reg_t*)recv_region[index];
    if(type == SEND_REGION)
        reg = (mp_reg_t*)send_region[index];
    if(type == SEND_STREAM_REGION)
        reg = (mp_reg_t*)send_stream_region[index];

    if(type == TIMER_RECV_REGION)
        reg = (mp_reg_t*)timer_recv_region[index];
    if(type == TIMER_SEND_REGION)
        reg = (mp_reg_t*)timer_send_region[index];

    assert(reg);

    if (!size) {
        ret = -EINVAL;
        comm_err("SIZE==0\n");
        goto out;
    }

#if 0
    //if (!*reg) {
        if(comm_rank == 0 && type == SEND_REQUEST)
           printf("Registering [%d] SEND buffer=%p, size: %d reg=%p\n", index, buf, size, reg);
    
        if(comm_rank == 0 && type == RECV_REQUEST)
           printf("Registering [%d] RECV buffer=%p, size: %d reg=%p\n", index, buf, size, reg);
        
        if(comm_rank == 0 && type == SEND_STREAM_REQUEST)
           printf("Registering [%d] SEND STREAM buffer=%p, size: %d reg=%p\n", index, buf, size, reg);
#endif
        DBG("registering buffer %p\n", buf);
        MP_CHECK(mp_register(buf, nbytes, reg));
    //}

out:
    return ret;
}




int comm_flush()
{
    int ret = 0;
    DBG("n_reqs=%d\n", n_reqs);
    assert(n_reqs < MAX_REQS);
#if 0
    do {
        rmb();
        uint32_t w0 = ACCESS_ONCE(ready_table[0]);
        uint32_t w1 = ACCESS_ONCE(ready_table[1]);
        DBG("ready_table: %08x %08x\n", w0, w1);
        ret = mp_progress_all(n_reqs, reqs);
        arch_cpu_relax();
        cudaStreamQuery(NULL);
    } while(ret < n_reqs);
#endif
    ret = mp_wait_all(n_reqs, reqs);
    if (ret) {
        comm_err("got error in mp_wait_all ret=%d\n", ret);
        exit(EXIT_FAILURE);
    }
    n_reqs = 0;
    return ret;
}

int comm_progress()
{
    int ret;
//    DBG("n_reqs=%d\n", n_reqs);
//    assert(n_reqs < MAX_REQS);
//    int ret = mp_progress_all(n_reqs, reqs);
    if( (80*startGlobalReqsIndex)+80 < n_reqs)
    {
        ret = mp_progress_all(80, reqs+startGlobalReqsIndex);
        if (ret < 0) {
            comm_err("ret=%d\n", ret);
        }
        startGlobalReqsIndex = (startGlobalReqsIndex+1)%MAX_REQS;
    }
   /*
    else
    {
        if(comm_rank == 0)
            printf("NO PROGRESS. n_reqs: %d, startGlobalReqsIndex: %d, (80*startGlobalReqsIndex)+80: %d\n",
                n_reqs, startGlobalReqsIndex, ((80*startGlobalReqsIndex)+80));
    }
    */
    return ret;
}


static struct comm_dev_descs *dreqs()
{
    if (!pdreqs) {
        cudaError_t err;
        err = cudaHostAlloc(&pdreqs, n_dreqs*sizeof(*pdreqs), 
                            cudaHostAllocPortable | cudaHostAllocMapped 
                            /*|cudaHostAllocWriteCombined*/ );
        if (err != cudaSuccess) {
            comm_err("error while allocating comm_dev_descs, exiting...\n");
            exit(-1);
        }
        assert(pdreqs);
        memset(pdreqs, 0, n_dreqs*sizeof(*pdreqs));
        dreq_idx = 0;
    }
    return pdreqs + dreq_idx;
}
    
int comm_prepare_wait_ready(int rank)
{
#if 0
    assert(comm_initialized);
    assert(rank < comm_size);
    int ret = 0;
    int peer = comm_mpi_rank_to_peer(rank);
    DBG("rank=%d payload=%x n_ready=%d\n", rank, ready_values[rank], dreqs()->n_ready);
    MP_CHECK(mp::mlx5::get_descriptors(&dreqs()->ready[dreqs()->n_ready++], &ready_table[rank], ready_values[rank]));
    //dreqs.ready.ptr = &ready_table[rank];
    //dreqs.ready.value = ready_values[rank];
    ready_values[rank]++;
    return ret;
#endif
}

int comm_prepare_isend(void *send_buf, size_t size, MPI_Datatype type, comm_reg_t *creg,
                       int dest_rank, comm_request_t *creq)
{
    assert(comm_initialized);
    int ret = 0;
    int retcode;
    size_t nbytes = size*comm_size_of_mpi_type(type);
    mp_reg_t *reg = (mp_reg_t*)creg;
    assert(reg);
    mp_request_t *req = (mp_request_t*)creq;
    int peer = comm_mpi_rank_to_peer(dest_rank);

    DBG("dest_rank=%d peer=%d nbytes=%d\n", dest_rank, peer, nbytes);

    if (!size) {
        ret = -EINVAL;
        comm_err("SIZE==0\n");
        goto err;
    }

    if (!*reg) {
        DBG("registering buffer %p\n", send_buf);
        MP_CHECK(mp_register(send_buf, nbytes, reg));
    }

    retcode = mp_send_prepare(send_buf, nbytes, peer, reg, req);
    if (retcode) {
        // BUG: call mp_unregister
        comm_err("error in mp_isend_on_stream ret=%d\n", retcode);
        ret = -1;
        goto unreg;
    }

    retcode = mp::mlx5::get_descriptors(&dreqs()->tx[dreqs()->n_tx++], req);
    if (retcode) {
        comm_err("error in mp_isend_on_stream ret=%d\n", retcode);
        ret = -1;
        goto unreg;
    }

    comm_track_request(req);

    return ret;

unreg:
    // BUG: call mp_unregister

err:
    return ret;
}

int comm_gpu_prepare_waitall(int count)
{
    int retcode;
    int ret = 0;
    
    assert(comm_initialized);
    mp_request_t * req = (mp_request_t *)&(timer_recv_requests[first_timer_recv_request]);
    for (int i=0; i < count; ++i) {
       /* if(comm_rank == 0)
            printf("comm_gpu_prepare_waitall, count=%d, first_timer_recv_request: %d i: %d, req=%p\n", 
                count, first_timer_recv_request, i, (req+i));
*/
        retcode = mp::mlx5::get_descriptors(&(pdreqs[indexGpuReqs].wait[i]), req+i);
        if (retcode) {
            comm_err("error in get_descriptors(wait) (%d)\n", retcode);
            ret = -1;
            goto out;
        }

        pdreqs[indexGpuReqs].n_wait++;
    }
    first_timer_recv_request += count;
    //memset(creqs, 0, sizeof(comm_request_t)*count);
out:
    return ret;
}

int comm_prepare_wait_all(int count, comm_request_t *creqs)
{
    int retcode;
    int ret = 0;
    DBG("count=%d\n", count);
    assert(comm_initialized);
    mp_request_t *req = (mp_request_t*)creqs;
    for (int i=0; i < count; ++i) {
        retcode = mp::mlx5::get_descriptors(&dreqs()->wait[dreqs()->n_wait++], req+i);
        if (retcode) {
            comm_err("error in get_descriptors(wait) (%d)\n", retcode);
            ret = -1;
            goto out;
        }
    }
    //memset(creqs, 0, sizeof(comm_request_t)*count);
out:
    return ret;
}

// Note: this is ugly and non thread-safe
comm_dev_descs_t comm_prepared_requests()
{
    comm_dev_descs_t ret;
    //static struct comm_dev_descs local;
    // copy dreqs to static storage
    //memcpy(&local, dreqs(), sizeof(local));
    // return static storage
    ret = dreqs();
    // super ugly
    dreq_idx = (dreq_idx + 1) % n_dreqs;
    // reset dreqs for next usage
    memset(dreqs(), 0, sizeof(struct comm_dev_descs));
    return ret;
}

//MPI_irecv
int comm_global_irecv(void *buf, int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm, MPI_Request *request, int index, cudaStream_t stream = NULL)
{
    int ret=0;
#if 0
    if(comm_rank == 6 && print == 1)
    {
       if(datatype == MPI_DOUBLE)
            printf("Recv from %d, index req: %d byte: %d sizeof(DOUBLE): %d\n",
             source, index, 8*count, sizeof(datatype));

        if(datatype == MPI_FLOAT)
            printf("Recv from %d, index req: %d byte: %d sizeof(FLOAT): %d\n",
             source, index, 4*count, sizeof(datatype));

    }
#endif

    if(comm_use_comm())
    {
        ret = comm_irecv_setup(buf, count, datatype, source, index);
    /*
        if(comm_use_async())
            comm_send_ready_stream_setup(source, stream);
        else
      */
            //if(!comm_use_async())
           // comm_send_ready_setup(source);
    }
    else
        ret = MPI_Irecv(buf, count, datatype, source, tag, comm, request);

    return ret;
}

int comm_gpu_irecv(void *recv_buf, int count,  MPI_Datatype type, int src_rank)
{
    int ret=0;
#if 0
    if(comm_rank == 0)
    {
       if(datatype == MPI_DOUBLE)
            printf("Recv from %d, index req: %d byte: %d sizeof(DOUBLE): %d\n",
             source, index, 8*count, sizeof(datatype));

        if(datatype == MPI_FLOAT)
            printf("Recv from %d, index req: %d byte: %d sizeof(FLOAT): %d\n",
             source, index, 4*count, sizeof(datatype));

    }
#endif

    if(comm_use_comm())
    {   
        indexTimerRecv = indexTimerRecv%MAX_REQS;
/*
        if(comm_rank == 0)
            printf("indexTimerRecv: %d, indexGpuReqs: %d src_rank: %d req=%p\n", 
                indexTimerRecv, indexGpuReqs, src_rank, &(timer_recv_requests[indexTimerRecv]));
*/
        comm_irecv(recv_buf, count, 
            type, timer_recv_region[0], src_rank, 
            &(timer_recv_requests[indexTimerRecv])
        );
        indexTimerRecv++;
    }
    
    return ret;
}

void comm_gpu_update_index()
{
    indexGpuReqs++;
}

struct comm_dev_descs comm_gpu_return_descr()
{
    return pdreqs[indexGpuReqs];
}

int comm_setup_gpu_prepare_isend(void *send_buf, int count, MPI_Datatype type, int dest_rank, int index)
{
    indexTimerSend = indexTimerSend%MAX_REQS;
 /*
    if(comm_rank == 0)
        printf("indexTimerSend: %d dest_rank: %d, index: %d\n", indexTimerSend, dest_rank, index);
 */
    comm_gpu_prepare_isend(send_buf, count, type, timer_send_region[0], dest_rank, &(timer_send_requests[indexTimerSend]), index);
    indexTimerSend++;

    return 0;
}

int comm_gpu_prepare_isend(void *send_buf, size_t size, MPI_Datatype type, comm_reg_t *creg,
 int dest_rank, comm_request_t *creq, int index)
{
    assert(comm_initialized);
    int ret = 0;
    int retcode;
    size_t nbytes = size*comm_size_of_mpi_type(type);
    mp_reg_t *reg = (mp_reg_t*)creg;
    assert(reg);
    mp_request_t *req = (mp_request_t*)creq;
    int peer = comm_mpi_rank_to_peer(dest_rank);
/*
    if(comm_rank == 0)
        printf("dest_rank=%d peer=%d nbytes=%d\n", dest_rank, peer, nbytes);
*/
    if (!size) {
        ret = -EINVAL;
        comm_err("SIZE==0\n");
        goto err;
    }

    if (!*reg) {
        DBG("registering buffer %p\n", send_buf);
        MP_CHECK(mp_register(send_buf, nbytes, reg));
    }

    retcode = mp_send_prepare(send_buf, nbytes, peer, reg, req);
    if (retcode) {
        // BUG: call mp_unregister
        comm_err("error in mp_isend_on_stream ret=%d\n", retcode);
        ret = -1;
        goto unreg;
    }

/*
    if(comm_rank == 0)
        printf("isend prepare req=%p, index: %d\n", req, index);
*/
    retcode = mp::mlx5::get_descriptors(&(pdreqs[indexGpuReqs].tx[index]), req);
    if (retcode) {
        comm_err("error in mp_isend_on_stream ret=%d\n", retcode);
        ret = -1;
        goto unreg;
    }

    pdreqs[indexGpuReqs].n_tx++;

    comm_track_request(req);

    return ret;

unreg:
    // BUG: call mp_unregister

err:
    return ret;
}

//MPI_Isend
int comm_global_isend(void *buf, int count, MPI_Datatype datatype, int dest,
    int tag, MPI_Comm comm, MPI_Request *request, int index)
{

#if 0
    if(comm_rank == 6 && print == 1)
    {
        if(datatype == MPI_DOUBLE)
            printf("Send to %d, index req: %d byte: %d sizeof(DOUBLE): %d\n",
             dest, index, 8*count, sizeof(datatype));

        if(datatype == MPI_FLOAT)
            printf("Send to %d, index req: %d byte: %d sizeof(FLOAT): %d\n",
             dest, index, 4*count, sizeof(datatype));

    }
#endif
    int ret=0;
    if(comm_use_comm())
    {
        //comm_wait_ready(dest);
        ret = comm_isend_setup(buf, count, datatype, dest, index);        
    }
    else
        ret = MPI_Isend(buf, count, datatype, dest, tag, comm, request);

    return ret;
}

//MPI_Wait
int comm_global_wait(MPI_Request *request, MPI_Status *status, int type, int index)
{
    int ret=0;
#if 0
    if(comm_rank == 6 && print == 1)
        printf("Wait type: %d index req: %d\n", type, index);
#endif
    if(comm_use_comm())
        ret = comm_wait_setup(type, index);
    else
        ret = MPI_Wait(request, status);

    return ret;
}

//MPI_Wait_all
int comm_global_wait_all(MPI_Request *request, MPI_Status *status, int count, int type)
{
    int ret=0;
    
    if(comm_use_comm())
        ret = comm_wait_all_setup(count, type);
    else
        ret = MPI_Waitall(count, request, status);

    return ret;
}

//MPI_Isend + stream
int comm_global_isend_stream(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, 
    MPI_Request *request, cudaStream_t stream, int index)

{
    int ret=0;
    if(comm_use_comm() && comm_use_async())
    {
        //comm_wait_ready_on_stream(dest, stream);
        ret = comm_isend_stream_setup(buf, count, datatype, dest, stream, index);
//        cudaDeviceSynchronize();
    }
    else if(comm_use_comm())
    {
        cudaStreamSynchronize(stream);
        //comm_wait_ready(dest);
        ret = comm_isend_setup(buf, count, datatype, dest, index);
    }
    else
    {
        cudaStreamSynchronize(stream);
        ret = MPI_Isend(buf, count, datatype, dest, tag, comm, request);
    }
        
    return ret;
}

//MPI_Wait + stream
int comm_global_wait_stream(MPI_Request *request, MPI_Status *status, int type, cudaStream_t stream, int index)
{
    int ret=0;

    if(comm_use_comm() && comm_use_async())
    {
        if(type == SEND_REQUEST || type == SEND_STREAM_REQUEST)
        {
            //cudaDeviceSynchronize();
            ret = comm_wait_stream_setup(SEND_STREAM_REQUEST, stream, index);
        }
            
        if(type == RECV_REQUEST)
        {
            //cudaDeviceSynchronize();
            ret = comm_wait_stream_setup(RECV_REQUEST, stream, index);
        }
    }
    else if(comm_use_comm())
        comm_wait_setup(type, index);
    else
        ret = MPI_Wait(request, status);

    return ret;
}

//MPI_Wait_all + stream
int comm_global_wait_all_stream(MPI_Request *request, MPI_Status *status, int count, int type, cudaStream_t stream = NULL)
{
    int ret=0;
    
    if(comm_use_comm() && comm_use_async())
    {
        if(type == SEND_REQUEST)
            ret = comm_wait_all_stream_setup(count, SEND_STREAM_REQUEST, stream);        
    }
    else if(comm_use_comm())
        comm_wait_all_setup(count, type);
    else
        ret = MPI_Waitall(count, request, status);

    return ret;\
}

void comm_setup_buf_maxsize(int comBufSize)
{
    maxBufSize = comBufSize;
}

void comm_register_step(void * buf, int type)
{   
    /*
    if(comm_rank == 6)
        printf("comm_register_step type %d, size: %d\n", type, maxBufSize);
    */
    if(type == RECV_REQUEST)
    {
        globalIndexRecvRegion++;
        comm_register(buf, maxBufSize, RECV_REQUEST);
    }
        
    if(type == SEND_REQUEST)
    {
        globalIndexSendRegion++;
        comm_register(buf, maxBufSize, SEND_REQUEST);
    }
        
}