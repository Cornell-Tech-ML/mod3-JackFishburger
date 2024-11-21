# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

Diagnostics Output:
The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

(mod3env) wy@dhcp-vl2051-136 mod3-JackFishburger % python project/parallel_check.py
MAP
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, /Users/wy/
Documents/cs5781_mle/workspace/mod3-JackFishburger/minitorch/fast_ops.py (164)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/wy/Documents/cs5781_mle/workspace/mod3-JackFishburger/minitorch/fast_ops.py (164) 
-------------------------------------------------------------------------|loop #ID
    def _map(                                                            | 
        out: Storage,                                                    | 
        out_shape: Shape,                                                | 
        out_strides: Strides,                                            | 
        in_storage: Storage,                                             | 
        in_shape: Shape,                                                 | 
        in_strides: Strides,                                             | 
    ) -> None:                                                           | 
        if (                                                             | 
            len(out_strides) == len(in_strides)                          | 
            and np.array_equal(out_strides, in_strides)                  | 
            and np.array_equal(out_shape, in_shape)                      | 
        ):                                                               | 
            for i in prange(out.size):-----------------------------------| #3
                out[i] = fn(in_storage[i])                               | 
            return                                                       | 
                                                                         | 
        size = np.prod(out_shape)----------------------------------------| #2
                                                                         | 
        for i in prange(size):-------------------------------------------| #4
            out_index = np.zeros(MAX_DIMS, np.int32)---------------------| #0
            in_index = np.zeros(MAX_DIMS, np.int32)----------------------| #1
                                                                         | 
            to_index(i, out_shape, out_index)                            | 
                                                                         | 
            broadcast_index(out_index, out_shape, in_shape, in_index)    | 
                                                                         | 
            in_pos = index_to_position(in_index, in_strides)             | 
            out_pos = index_to_position(out_index, out_strides)          | 
                                                                         | 
            out[out_pos] = fn(in_storage[in_pos])                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #3, #2, #4, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--4 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--4 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--4 (parallel)
   +--0 (serial, fused with loop(s): 1)


 
Parallel region 0 (loop #4) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#4).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/wy/Documents/cs5781
_mle/workspace/mod3-JackFishburger/minitorch/fast_ops.py (184) is hoisted out of
 the parallel loop labelled #4 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/wy/Documents/cs5781
_mle/workspace/mod3-JackFishburger/minitorch/fast_ops.py (185) is hoisted out of
 the parallel loop labelled #4 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, /Users/wy/
Documents/cs5781_mle/workspace/mod3-JackFishburger/minitorch/fast_ops.py (222)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/wy/Documents/cs5781_mle/workspace/mod3-JackFishburger/minitorch/fast_ops.py (222) 
-----------------------------------------------------------------------|loop #ID
    def _zip(                                                          | 
        out: Storage,                                                  | 
        out_shape: Shape,                                              | 
        out_strides: Strides,                                          | 
        a_storage: Storage,                                            | 
        a_shape: Shape,                                                | 
        a_strides: Strides,                                            | 
        b_storage: Storage,                                            | 
        b_shape: Shape,                                                | 
        b_strides: Strides,                                            | 
    ) -> None:                                                         | 
        if (                                                           | 
            len(out_strides) == len(a_strides) == len(b_strides)       | 
            and np.array_equal(out_strides, a_strides)                 | 
            and np.array_equal(out_strides, b_strides)                 | 
            and np.array_equal(out_shape, a_shape)                     | 
            and np.array_equal(out_shape, b_shape)                     | 
        ):                                                             | 
            for i in prange(out.size):---------------------------------| #9
                out[i] = fn(a_storage[i], b_storage[i])                | 
            return                                                     | 
                                                                       | 
        size = np.prod(out_shape)--------------------------------------| #8
                                                                       | 
        for i in prange(size):-----------------------------------------| #10
            out_index = np.zeros(MAX_DIMS, np.int32)-------------------| #5
            a_index = np.zeros(MAX_DIMS, np.int32)---------------------| #6
            b_index = np.zeros(MAX_DIMS, np.int32)---------------------| #7
                                                                       | 
            to_index(i, out_shape, out_index)                          | 
                                                                       | 
            broadcast_index(out_index, out_shape, a_shape, a_index)    | 
            broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                                                                       | 
            a_pos = index_to_position(a_index, a_strides)              | 
            b_pos = index_to_position(b_index, b_strides)              | 
            out_pos = index_to_position(out_index, out_strides)        | 
                                                                       | 
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--5 has the following loops fused into it:
   +--6 (fused)
   +--7 (fused)
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #9, #8, #10, #5).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--10 is a parallel loop
   +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--5 (parallel)
   +--6 (parallel)
   +--7 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--5 (serial, fused with loop(s): 6, 7)


 
Parallel region 0 (loop #10) had 2 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/wy/Documents/cs5781
_mle/workspace/mod3-JackFishburger/minitorch/fast_ops.py (247) is hoisted out of
 the parallel loop labelled #10 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/wy/Documents/cs5781
_mle/workspace/mod3-JackFishburger/minitorch/fast_ops.py (248) is hoisted out of
 the parallel loop labelled #10 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/wy/Documents/cs5781
_mle/workspace/mod3-JackFishburger/minitorch/fast_ops.py (249) is hoisted out of
 the parallel loop labelled #10 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: b_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, /Use
rs/wy/Documents/cs5781_mle/workspace/mod3-JackFishburger/minitorch/fast_ops.py 
(286)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/wy/Documents/cs5781_mle/workspace/mod3-JackFishburger/minitorch/fast_ops.py (286) 
--------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                | 
        out: Storage,                                                           | 
        out_shape: Shape,                                                       | 
        out_strides: Strides,                                                   | 
        a_storage: Storage,                                                     | 
        a_shape: Shape,                                                         | 
        a_strides: Strides,                                                     | 
        reduce_dim: int,                                                        | 
    ) -> None:                                                                  | 
        size = np.prod(out_shape)-----------------------------------------------| #12
        reduce_size = a_shape[reduce_dim]                                       | 
        for i in prange(size):--------------------------------------------------| #13
            index = np.zeros(MAX_DIMS, np.int32)--------------------------------| #11
            to_index(i, out_shape, index)                                       | 
            out_pos = index_to_position(index, out_strides)                     | 
            index[reduce_dim] = 0                                               | 
            in_pos = index_to_position(index, a_strides)                        | 
            accumulated_value = a_storage[in_pos]                               | 
            for j in range(1, reduce_size):                                     | 
                index[reduce_dim] = j                                           | 
                in_pos = index_to_position(index, a_strides)                    | 
                accumulated_value = fn(accumulated_value, a_storage[in_pos])    | 
            out[out_pos] = accumulated_value                                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #12, #13, #11).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--13 is a parallel loop
   +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--11 (serial)


 
Parallel region 0 (loop #13) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/wy/Documents/cs5781
_mle/workspace/mod3-JackFishburger/minitorch/fast_ops.py (298) is hoisted out of
 the parallel loop labelled #13 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, /Users/wy/D
ocuments/cs5781_mle/workspace/mod3-JackFishburger/minitorch/fast_ops.py (313)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/wy/Documents/cs5781_mle/workspace/mod3-JackFishburger/minitorch/fast_ops.py (313) 
----------------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                              | 
    out: Storage,                                                                                         | 
    out_shape: Shape,                                                                                     | 
    out_strides: Strides,                                                                                 | 
    a_storage: Storage,                                                                                   | 
    a_shape: Shape,                                                                                       | 
    a_strides: Strides,                                                                                   | 
    b_storage: Storage,                                                                                   | 
    b_shape: Shape,                                                                                       | 
    b_strides: Strides,                                                                                   | 
) -> None:                                                                                                | 
    """NUMBA tensor matrix multiply function.                                                             | 
                                                                                                          | 
    Should work for any tensor shapes that broadcast as long as                                           | 
                                                                                                          | 
    ```                                                                                                   | 
    assert a_shape[-1] == b_shape[-2]                                                                     | 
    ```                                                                                                   | 
                                                                                                          | 
    Optimizations:                                                                                        | 
                                                                                                          | 
    * Outer loop in parallel                                                                              | 
    * No index buffers or function calls                                                                  | 
    * Inner loop should have no global writes, 1 multiply.                                                | 
                                                                                                          | 
                                                                                                          | 
    Args:                                                                                                 | 
    ----                                                                                                  | 
        out (Storage): storage for `out` tensor                                                           | 
        out_shape (Shape): shape for `out` tensor                                                         | 
        out_strides (Strides): strides for `out` tensor                                                   | 
        a_storage (Storage): storage for `a` tensor                                                       | 
        a_shape (Shape): shape for `a` tensor                                                             | 
        a_strides (Strides): strides for `a` tensor                                                       | 
        b_storage (Storage): storage for `b` tensor                                                       | 
        b_shape (Shape): shape for `b` tensor                                                             | 
        b_strides (Strides): strides for `b` tensor                                                       | 
                                                                                                          | 
    Returns:                                                                                              | 
    -------                                                                                               | 
        None : Fills in `out`                                                                             | 
                                                                                                          | 
    """                                                                                                   | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                                | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                                | 
                                                                                                          | 
    a_col_stride = a_strides[1]                                                                           | 
    a_row_stride = a_strides[2]                                                                           | 
                                                                                                          | 
    b_col_stride = b_strides[1]                                                                           | 
    b_row_stride = b_strides[2]                                                                           | 
                                                                                                          | 
    result_dim = b_shape[-2]                                                                              | 
                                                                                                          | 
    for batch_index in prange(out_shape[0]):--------------------------------------------------------------| #14
        for row in range(out_shape[1]):                                                                   | 
            for col in range(out_shape[2]):                                                               | 
                a_index = batch_index * a_batch_stride + row * a_col_stride                               | 
                b_index = batch_index * b_batch_stride + col * b_row_stride                               | 
                out_index = batch_index * out_strides[0] + row * out_strides[1] + col * out_strides[2]    | 
                result = 0.0                                                                              | 
                for _ in range(result_dim):                                                               | 
                    result += a_storage[a_index] * b_storage[b_index]                                     | 
                    a_index += a_row_stride                                                               | 
                    b_index += b_col_stride                                                               | 
                out[out_index] = result                                                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #14).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None