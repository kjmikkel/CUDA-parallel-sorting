#ifndef PARACUDA_STRUCT_H
#define PARACUDA_STRUCT_H

#define PARACUDA_SINGLE(T0) \
typedef T0 PARACUDA_##T0##_struct; \
__inline__ __device__ \
void PARACUDA_##T0##_from_vector(T0* v, T0* r, size_t index) \
{ \
    *v = r[index]; \
} \
void PARACUDA_##T0##_from_vector_host(T0* v, T0* r, size_t index) \
{ \
    *v = r[index]; \
} \
__inline__ __device__ \
void PARACUDA_##T0##_to_vector(T0* r, T0* v, size_t index) \
{ \
  r[index] = *v;				\
} \
/* TODO: eliminate this */ \
void PARACUDA_##T0##_to_vector_poke(T0* r, T0* v, size_t index) \
{ \
    cutilSafeCall(cudaMemcpy(&(r[index]), v, sizeof(T0), cudaMemcpyHostToDevice)); \
} \
void PARACUDA_##T0##_to_vector_host(T0* r, T0* v, size_t index) \
{ \
    r[index] = *v; \
} \
T0* PARACUDA_##T0##_allocate_host(size_t length) \
{ \
    T0* r = (T0*) malloc(length * sizeof(T0)); \
    return r; \
} \
T0* PARACUDA_##T0##_allocate_device(size_t length) \
{ \
  T0* r ;  \
  cutilSafeCall(cudaMalloc((void**) &(r), length * sizeof(T0)));\
  return r; \
} \
void PARACUDA_##T0##_free_device(T0* r) \
{ \
    cutilSafeCall(cudaFree(r)); \
} \
void PARACUDA_##T0##_free_host(T0* r) \
{ \
    free(r); \
} \
void PARACUDA_##T0##_copy_host_device(T0* out, T0* in, size_t length) \
{ \
    cutilSafeCall(cudaMemcpy(out, in, length * sizeof(T0), cudaMemcpyHostToDevice)); \
} \
void PARACUDA_##T0##_copy_device_host(T0* out, T0* in, size_t length) \
{ \
    cutilSafeCall(cudaMemcpy(out, in, length * sizeof(T0), cudaMemcpyDeviceToHost)); \
} \
int PARACUDA_##T0##_equal(T0* a, int ai, T0* b, int bi) \
{ \
    return a[ai] == b[bi]; \
} \


#define PARACUDA_STRUCT_2(NAME, VECTOR, T0, N0, T1, N1) \
typedef struct NAME NAME; \
struct NAME { T0 N0; T1 N1; }; \
typedef struct VECTOR PARACUDA_##NAME##_struct; \
typedef struct VECTOR VECTOR; \
struct VECTOR { T0* N0; T1* N1; }; \
__inline__ __device__ \
void PARACUDA_##NAME##_from_vector(NAME* v, VECTOR* r, size_t index) \
{ \
    v->N0 = r->N0[index]; \
    v->N1 = r->N1[index]; \
} \
void PARACUDA_##NAME##_from_vector_host(NAME* v, VECTOR* r, size_t index) \
{ \
    v->N0 = r->N0[index]; \
    v->N1 = r->N1[index]; \
} \
__inline__ __device__ \
void PARACUDA_##NAME##_to_vector(VECTOR* r, NAME* v, size_t index) \
{ \
    r->N0[index] = v->N0; \
    r->N1[index] = v->N1; \
} \
void PARACUDA_##NAME##_to_vector_host(VECTOR* r, NAME* v, size_t index) \
{ \
    r->N0[index] = v->N0; \
    r->N1[index] = v->N1; \
} \
VECTOR* PARACUDA_##NAME##_allocate_host(size_t length) \
{ \
    VECTOR* r = (VECTOR*) malloc(sizeof(VECTOR)); \
    r->N0 = (T0*) malloc(length * sizeof(T0)); \
    r->N1 = (T1*) malloc(length * sizeof(T1)); \
    return r; \
} \
VECTOR* PARACUDA_##NAME##_allocate_device(size_t length) \
{ \
    VECTOR t; \
    cutilSafeCall(cudaMalloc((void**) &(t.N0), length * sizeof(T0))); \
    cutilSafeCall(cudaMalloc((void**) &(t.N1), length * sizeof(T1))); \
    VECTOR* r; \
    cutilSafeCall(cudaMalloc((void**) &r, sizeof(VECTOR))); \
    cutilSafeCall(cudaMemcpy(r, &t, sizeof(VECTOR), cudaMemcpyHostToDevice)); \
    return r; \
} \
void PARACUDA_##NAME##_free_device(VECTOR* r) \
{ \
    VECTOR t; \
    cutilSafeCall(cudaMemcpy(&t, r, sizeof(VECTOR), cudaMemcpyDeviceToHost)); \
    cutilSafeCall(cudaFree(t.N0)); \
    cutilSafeCall(cudaFree(t.N1)); \
    cutilSafeCall(cudaFree(r)); \
} \
void PARACUDA_##NAME##_free_host(VECTOR* r) \
{ \
    free(r->N0); \
    free(r->N1); \
    free(r); \
} \
void PARACUDA_##NAME##_copy_host_device(VECTOR* out, VECTOR* in, size_t length) \
{ \
    VECTOR temp; \
    cutilSafeCall(cudaMemcpy(&temp, out, sizeof(VECTOR), cudaMemcpyDeviceToHost)); \
    cutilSafeCall(cudaMemcpy(temp.N0, in->N0, length * sizeof(T0), cudaMemcpyHostToDevice)); \
    cutilSafeCall(cudaMemcpy(temp.N1, in->N1, length * sizeof(T1), cudaMemcpyHostToDevice)); \
} \
void PARACUDA_##NAME##_copy_device_host(VECTOR* out, VECTOR* in, size_t length) \
{ \
    VECTOR temp; \
    cutilSafeCall(cudaMemcpy(&temp, in, sizeof(VECTOR), cudaMemcpyDeviceToHost));	      \
    cutilSafeCall(cudaMemcpy(out->N0, temp.N0, length * sizeof(T0), cudaMemcpyDeviceToHost)); \
    cutilSafeCall(cudaMemcpy(out->N1, temp.N1, length * sizeof(T1), cudaMemcpyDeviceToHost)); \
} \
/* TODO: eliminate this */ \
void PARACUDA_##NAME##_to_vector_poke(VECTOR* r, NAME* v, size_t index) \
{ \
    cutilSafeCall(cudaMemcpy(&(r->N0[index]), &(v->N0), sizeof(T0), cudaMemcpyHostToDevice)); \
    cutilSafeCall(cudaMemcpy(&(r->N1[index]), &(v->N1), sizeof(T1), cudaMemcpyHostToDevice)); \
} \
int PARACUDA_##NAME##_equal(VECTOR* a, int ai, VECTOR* b, int bi) \
{ \
    return a->N0[ai] == b->N0[bi] && a->N1[ai] == b->N1[bi]; \
} \


#define PARACUDA_STRUCT_3(NAME, VECTOR, T0, N0, T1, N1, T2, N2)	\
typedef struct NAME NAME; \
struct NAME { T0 N0; T1 N1; T2 N2; };	\
typedef struct VECTOR PARACUDA_##NAME##_struct; \
typedef struct VECTOR VECTOR; \
struct VECTOR { T0* N0; T1* N1; T2* N2; };	\
__inline__ __device__ \
void PARACUDA_##NAME##_from_vector(NAME* v, VECTOR* r, size_t index) \
{ \
    v->N0 = r->N0[index]; \
    v->N1 = r->N1[index]; \
    v->N2 = r->N2[index]; \
} \
void PARACUDA_##NAME##_from_vector_host(NAME* v, VECTOR* r, size_t index) \
{ \
    v->N0 = r->N0[index]; \
    v->N1 = r->N1[index]; \
    v->N2 = r->N2[index]; \
} \
__inline__ __device__ \
void PARACUDA_##NAME##_to_vector(VECTOR* r, NAME* v, size_t index) \
{ \
    r->N0[index] = v->N0; \
    r->N1[index] = v->N1; \
    r->N2[index] = v->N2; \
} \
void PARACUDA_##NAME##_to_vector_host(VECTOR* r, NAME* v, size_t index) \
{ \
    r->N0[index] = v->N0; \
    r->N1[index] = v->N1; \
    r->N2[index] = v->N2; \
} \
VECTOR* PARACUDA_##NAME##_allocate_host(size_t length) \
{ \
    VECTOR* r = (VECTOR*) malloc(sizeof(VECTOR)); \
    r->N0 = (T0*) malloc(length * sizeof(T0)); \
    r->N1 = (T1*) malloc(length * sizeof(T1)); \
    r->N2 = (T2*) malloc(length * sizeof(T2)); \
    return r; \
} \
VECTOR* PARACUDA_##NAME##_allocate_device(size_t length) \
{ \
    VECTOR t; \
    cutilSafeCall(cudaMalloc((void**) &(t.N0), length * sizeof(T0))); \
    cutilSafeCall(cudaMalloc((void**) &(t.N1), length * sizeof(T1))); \
    cutilSafeCall(cudaMalloc((void**) &(t.N2), length * sizeof(T2))); \
    VECTOR* r; \
    cutilSafeCall(cudaMalloc((void**) &r, sizeof(VECTOR))); \
    cutilSafeCall(cudaMemcpy(r, &t, sizeof(VECTOR), cudaMemcpyHostToDevice)); \
    return r; \
} \
void PARACUDA_##NAME##_free_device(VECTOR* r) \
{ \
    VECTOR t; \
    cutilSafeCall(cudaMemcpy(&t, r, sizeof(VECTOR), cudaMemcpyDeviceToHost)); \
    cutilSafeCall(cudaFree(t.N0)); \
    cutilSafeCall(cudaFree(t.N1)); \
    cutilSafeCall(cudaFree(t.N2)); \
    cutilSafeCall(cudaFree(r)); \
} \
void PARACUDA_##NAME##_free_host(VECTOR* r) \
{ \
    free(r->N0); \
    free(r->N1); \
    free(r->N2); \
    free(r); \
} \
void PARACUDA_##NAME##_copy_host_device(VECTOR* out, VECTOR* in, size_t length) \
{ \
    VECTOR temp; \
    cutilSafeCall(cudaMemcpy(&temp, out, sizeof(VECTOR), cudaMemcpyDeviceToHost)); \
    cutilSafeCall(cudaMemcpy(temp.N0, in->N0, length * sizeof(T0), cudaMemcpyHostToDevice)); \
    cutilSafeCall(cudaMemcpy(temp.N1, in->N1, length * sizeof(T1), cudaMemcpyHostToDevice)); \
    cutilSafeCall(cudaMemcpy(temp.N2, in->N2, length * sizeof(T2), cudaMemcpyHostToDevice)); \
} \
void PARACUDA_##NAME##_copy_device_host(VECTOR* out, VECTOR* in, size_t length) \
{ \
    VECTOR temp; \
    cutilSafeCall(cudaMemcpy(&temp, in, sizeof(VECTOR), cudaMemcpyDeviceToHost));	      \
    cutilSafeCall(cudaMemcpy(out->N0, temp.N0, length * sizeof(T0), cudaMemcpyDeviceToHost)); \
    cutilSafeCall(cudaMemcpy(out->N1, temp.N1, length * sizeof(T1), cudaMemcpyDeviceToHost)); \
    cutilSafeCall(cudaMemcpy(out->N2, temp.N2, length * sizeof(T2), cudaMemcpyDeviceToHost)); \
} \
int PARACUDA_##NAME##_equal(VECTOR* a, int ai, VECTOR* b, int bi) \
{ \
    return a->N0[ai] == b->N0[bi] && a->N1[ai] == b->N1[bi] && a->N2[ai] == b->N2[bi]; \
} \


#endif
