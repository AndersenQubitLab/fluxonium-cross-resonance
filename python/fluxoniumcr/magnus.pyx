cimport cython
from libc.math cimport sin, cos
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from scipy.linalg.cython_lapack cimport zheev
from scipy.linalg.cython_blas cimport zgemm

from collections import namedtuple

import numpy as np


OdeResult = namedtuple('OdeResult', 't y')

cdef int MAGNUS_ORDER = 3
cdef double complex COMPLEX_ONE = 1.0
cdef double complex COMPLEX_ZERO = 0.0


cdef struct EigenvectorPropagatorCache:
    int N
    double *W
    double *RWORK
    double complex *WORK
    int LWORK

    double complex *REG0
    double complex *REG1


@cython.boundscheck(False)
@cython.wraparound(False)
def sesolve_magnusgl6(
        H not None,
        psi0 not None,
        tlist not None,
):
    # General purpose variables.
    cdef int i, j, k
    cdef double dt
    cdef const double complex[:, :] memview

    cdef int num_rows

    if hasattr(H[0], 'astype'):
        # H[0] is a constant operator
        # (If H[0] has an 'astype' attribute, it is probably an ArrayLike.)
        num_rows = H[0].shape[0]
    else:
        # H[0] is an (operator, function) pair
        num_rows = H[0][0].shape[0]

    cdef int num_cols = psi0.shape[1]
    cdef int num_ops = len(H)
    cdef int num_times = len(tlist)
    cdef int ham_size = num_rows * num_rows
    cdef int state_size = num_rows * num_cols

    if num_cols > num_rows:
        raise ValueError(
            "psi0 cannot have more columns than rows, got psi0.shape=" + str(psi0.shape),
        )

    states = np.empty((num_times, num_cols, num_rows), dtype=complex)
    cdef double complex[:, :, :] states_view = states
    states[0] = psi0.T

    cdef void *ops_ptr = malloc(num_ops*ham_size * sizeof(double complex))
    cdef void *c_ptr = malloc(num_ops*MAGNUS_ORDER*(num_times-1) * sizeof(double complex))

    cdef double complex[:, :, :]ops = \
        <double complex[:num_ops:1, :num_rows, :num_rows]>ops_ptr
    cdef double complex[:, :, :]c = \
        <double complex[:num_ops:1, :MAGNUS_ORDER, :num_times-1]>c_ptr

    eval_times = np.empty((MAGNUS_ORDER, num_times - 1), dtype=float, order='F')
    cdef double[::1, :] eval_times_view = eval_times
    cdef const double[::1] tlist_view = tlist
    for i in range(num_times - 1):
        dt = tlist_view[i+1] - tlist_view[i]
        eval_times_view[0, i] = tlist_view[i] + 0.1127016653792583 * dt  # t_i + (0.5 - sqrt(15)/10)*dt
        eval_times_view[1, i] = tlist_view[i] + 0.5 * dt
        eval_times_view[2, i] = tlist_view[i] + 0.8872983346207417 * dt  # t_i + (0.5 + sqrt(15)/10)*dt

    for i in range(num_ops):
        if hasattr(H[i], 'astype'):
            # H[i] is a constant operator
            # (If H[i] has an 'astype' attribute, it is probably an ArrayLike.)
            for j in range(MAGNUS_ORDER):
                for k in range(num_times-1):
                    c[i, j, k] = 1.0
            memview = H[i].astype(complex, copy=False)
        else:
            # H[i] is an (operator, function) pair
            func = H[i][1]
            memview = func(eval_times).astype(complex, copy=False)
            for j in range(MAGNUS_ORDER):
                for k in range(num_times-1):
                    c[i, j, k] = memview[j, k]
            memview = H[i][0].astype(complex, copy=False)

        for j in range(num_rows):
            for k in range(num_rows):
                ops[i, j, k] = memview[j, k]

    cdef EigenvectorPropagatorCache cache
    cdef int cache_init = 0

    cdef void *Heval_ptr = malloc(ham_size*MAGNUS_ORDER * sizeof(double complex))
    cdef void *G_ptr = malloc(ham_size*5 * sizeof(double complex))
    cdef double complex[::1, :, :] Heval = \
        <double complex[:num_rows:1, :num_rows, :MAGNUS_ORDER]>Heval_ptr
    cdef double complex[::1, :, :] G = \
        <double complex[:num_rows:1, :num_rows, :5]>G_ptr

    for i in range(num_times - 1):
        dt = tlist_view[i+1] - tlist_view[i]

        zgemm(
            'T',
            'N',
            &ham_size,
            &MAGNUS_ORDER,
            &num_ops,
            &COMPLEX_ONE,
            &ops[0, 0, 0],
            &num_ops,
            &c[0, 0, i],
            &num_ops,
            &COMPLEX_ZERO,
            &Heval[0, 0, 0],
            &ham_size,
        )

        for k in range(num_rows):
            for j in range(num_rows):
                G[j, k, 1] = 1.2909944487358056 * (Heval[j, k, 2] - Heval[j, k, 0])
                G[j, k, 2] = (
                    3.3333333333333335 * (Heval[j, k, 2] + Heval[j, k, 0])
                     - 6.666666666666667 * Heval[j, k, 1]
                )
                G[j, k, 0] = Heval[j, k, 1] + 0.08333333333333333 * G[j, k, 2]

        memcpy(&G[0, 0, 3], &G[0, 0, 1], 2 * ham_size * sizeof(double complex))

        commutator(
            num_rows,
            -1j * dt,
            &Heval[0, 0, 1],
            &G[0, 0, 1],
            2.0,
            &G[0, 0, 4],
        )

        commutator(
            num_rows,
            0.016666666666666666j * dt,
            &Heval[0, 0, 1],
            &G[0, 0, 4],
            1.0,
            &G[0, 0, 3],
        )

        for k in range(num_rows):
            for j in range(num_rows):
                G[j, k, 4] -= 20*Heval[j, k, 1] + 3*G[j, k, 2]

        commutator(
            num_rows,
            0.004166666666666667j * dt,
            &G[0, 0, 4],
            &G[0, 0, 3],
            -1.0,
            &G[0, 0, 0],
        )

        if cache_init == 0:
            alloc_cache(&cache, num_rows, &G[0, 0, 0])
            cache_init = 1

        propagate(
            num_cols,
            dt,
            &G[0, 0, 0],
            &states_view[i, 0, 0],
            &states_view[i+1, 0, 0],
            &cache,
        )

    free(ops_ptr)
    free(c_ptr)
    free(Heval_ptr)
    free(G_ptr)
    free_cache(&cache)

    return OdeResult(
        t=tlist.copy(),
        y=states.swapaxes(-2, -1),
    )


cdef commutator(
        int N,
        double complex alpha,
        double complex *A,
        double complex *B,
        double complex beta,
        double complex *C,
):
    zgemm('N', 'N', &N, &N, &N, &alpha, A, &N, B, &N, &beta, C, &N)
    alpha = -alpha
    zgemm('N', 'N', &N, &N, &N, &alpha, B, &N, A, &N, &COMPLEX_ONE, C, &N)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef alloc_cache(
        EigenvectorPropagatorCache *cache,
        int N,
        double complex *G,
):
    cache.N = N
    cache.W = <double *>malloc(sizeof(double) * N)
    cache.RWORK = <double *>malloc(sizeof(double) * (3*N - 2))

    cache.REG0 = <double complex *>malloc(sizeof(double complex) * N)
    cache.REG1 = <double complex *>malloc(sizeof(double complex) * N * N)

    cdef double complex query
    cdef int info

    cache.LWORK = -1

    zheev(
        'V',
        'L',
        &N,
        G,
        &N,
        cache.W,
        &query,
        &cache.LWORK,
        cache.RWORK,
        &info,
    )

    cache.LWORK = <int>query.real
    cache.WORK = <double complex *>malloc(sizeof(double complex) * cache.LWORK)


cdef free_cache(EigenvectorPropagatorCache *cache):
    free(cache.WORK)
    free(cache.W)
    free(cache.RWORK)
    free(cache.REG0)
    free(cache.REG1)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef propagate(
        int num_cols,
        double t,
        double complex *G,
        double complex *B,
        double complex *C,
        EigenvectorPropagatorCache *cache,
):
    cdef int i, j, info, offset
    zheev(
        'V',
        'L',
        &cache.N,
        G,
        &cache.N,
        cache.W,
        cache.WORK,
        &cache.LWORK,
        cache.RWORK,
        &info,
    )

    for i in range(cache.N):
        cache.REG0[i].real = cos(t * cache.W[i])
        cache.REG0[i].imag = sin(t * cache.W[i])

    cdef double complex alpha = 1.0
    cdef double complex beta = 0.0

    zgemm(
        'C',
        'N',
        &cache.N,
        &num_cols,
        &cache.N,
        &alpha,
        G,  # Recall G now contains eigenvectors in its columns.
        &cache.N,
        B,
        &cache.N,
        &beta,
        cache.REG1,
        &cache.N
    )

    for j in range(num_cols):
        offset = j*cache.N
        for i in range(cache.N):
            cache.REG1[offset + i] *= cache.REG0[i]

    zgemm(
        'N',
        'N',
        &cache.N,
        &num_cols,
        &cache.N,
        &alpha,
        G,
        &cache.N,
        cache.REG1,
        &cache.N,
        &beta,
        C,
        &cache.N,
    )
