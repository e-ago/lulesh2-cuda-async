#pragma once

#ifdef __cplusplus
#include <mpi.h>
#endif

#define __COMM_CHECK(stmt, cond_str)                    \
    do {                                \
            int result = (stmt);                                        \
            if (result) {                                               \
                fprintf(stderr, "%s failed at %s:%d error=%d\n",        \
                        cond_str, __FILE__, __LINE__, result);          \
                exit(EXIT_FAILURE);                                     \
            }                                                           \
        } while (0)

#define COMM_CHECK(stmt) __COMM_CHECK(stmt, #stmt)

#define RECV_REQUEST 0
#define SEND_REQUEST 1
#define SEND_STREAM_REQUEST 2        
#define READY_REQUEST 3

#define TIMER_RECV_REQUEST 4
#define TIMER_SEND_REQUEST 5

#define RECV_REGION 0
#define SEND_REGION 1
#define SEND_STREAM_REGION 2


#define TIMER_RECV_REGION 4
#define TIMER_SEND_REGION 5

#define MAX_REQS 65536


#ifdef __cplusplus
extern "C" {
#endif

    int comm_use_comm();
    int comm_use_gdrdma();
    int comm_use_async();
    int comm_use_gpu_comm();

    typedef struct comm_request  *comm_request_t;
    typedef struct comm_reg      *comm_reg_t;
    typedef struct CUstream_st   *comm_stream_t;
    int comm_init(MPI_Comm comm);
    void comm_finalize();
    int comm_send_ready_on_stream(int rank, comm_request_t *creq, comm_stream_t stream);
    int comm_send_ready(int rank, comm_request_t *creq);
    int comm_wait_ready_on_stream(int rank, comm_stream_t stream);
    int comm_wait_ready(int rank);
    int comm_test_ready(int rank, int *p_rdy);

    int comm_regions_setup(int numReq, int type);
    int comm_requests_setup(int numReq, int type);

    int comm_reset(int type);

    int comm_irecv(void *recv_buf, size_t size, MPI_Datatype type, comm_reg_t *reg, int src_rank, 
                   comm_request_t *req);

    int comm_irecv_setup(void *recv_buf, size_t size, MPI_Datatype type, int src_rank, int index);

    int comm_isend_on_stream(void *send_buf, size_t size, MPI_Datatype type, comm_reg_t *reg,
                             int dest_rank, comm_request_t *req, comm_stream_t stream);
    int comm_isend(void *send_buf, size_t size, MPI_Datatype type, comm_reg_t *reg,
                   int dest_rank, comm_request_t *req);
    int comm_isend_stream_setup(void *send_buf, size_t size, MPI_Datatype type, int dest_rank, cudaStream_t stream, int index);

    int comm_global_isend_stream(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request, cudaStream_t stream, int index);

    int comm_isend_setup(void *send_buf, size_t size, MPI_Datatype type, int dest_rank);

    int comm_wait_all_setup(int count, int type);

    int comm_wait_all(int count, comm_request_t *creqs);
    
    int comm_wait_all_stream_setup(int type, cudaStream_t stream);
    int comm_wait_all_on_stream(int count, comm_request_t *creqs, comm_stream_t stream);
    int comm_global_wait_all_stream(MPI_Request *request, MPI_Status *status, int count, int type, cudaStream_t stream);
    int comm_global_wait_all(MPI_Request *request, MPI_Status *status, int count, int type);
    int comm_global_wait(MPI_Request *request, MPI_Status *status, int type, int index);

    int comm_wait_stream_setup(int type, cudaStream_t stream, int index);

    int comm_global_wait_stream(MPI_Request *request, MPI_Status *status, int type, cudaStream_t stream, int index);
    int comm_wait_setup(int type, int index);
    int comm_wait(comm_request_t *creq);

    int comm_flush();
    int comm_progress();

    int comm_prepare_wait_ready(int rank);
    int comm_prepare_isend(void *send_buf, size_t size, MPI_Datatype type, comm_reg_t *creg,
                           int dest_rank, comm_request_t *creq);
    int comm_prepare_wait_all(int count, comm_request_t *creqs);
    int comm_register(void *buf, size_t size, int type);

    int comm_global_irecv(void *buf, int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm, MPI_Request *request, int index, cudaStream_t stream);
    int comm_global_isend(void *buf, int count, MPI_Datatype datatype, int dest,
        int tag, MPI_Comm comm, MPI_Request *request, int index);
    

    double comm_wait_put_value(int rank, double myValue);
    double comm_min_op_put_value(int numRank, int myRank, double myValue);
    int comm_put_value(int rank, double *valueToSend, comm_request_t *creq);
    void comm_cleanup_put_table();

    void comm_setup_buf_maxsize(int comBufSize);
    void comm_register_step(void * buf, int type);

    int comm_send_ready_setup(int rank);
    int comm_send_ready_stream_setup(int rank, comm_request_t *creq, cudaStream_t stream);
    int comm_register_index(void *buf, size_t size, int type, int index);

#ifdef __cplusplus
}
#endif
