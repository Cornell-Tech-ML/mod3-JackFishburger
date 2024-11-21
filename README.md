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

TASK 3.2 and 3.3 Diagnostics Output:
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

Task 3.5:
Simple:
!cd $DIR; PYTHONPATH=/content/$DIR python3.11 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET simple --RATE 0.05
Epoch  0  loss  5.9389108674948705 correct 43 time 24.031527996063232
Epoch  10  loss  0.6189816983642062 correct 50 time 0.20042943954467773
Epoch  20  loss  1.4568134493438023 correct 49 time 0.2285919189453125
Epoch  30  loss  0.7420451975928736 correct 49 time 0.40831518173217773
Epoch  40  loss  0.6893048793950689 correct 50 time 0.21571850776672363
Epoch  50  loss  0.17241009284365622 correct 50 time 0.19939947128295898
Epoch  60  loss  0.05265437489545163 correct 50 time 0.20034551620483398
Epoch  70  loss  0.22328968688020576 correct 50 time 0.19799208641052246
Epoch  80  loss  0.5189978639569454 correct 50 time 0.19952392578125
Epoch  90  loss  0.14608489612990408 correct 50 time 0.3318638801574707
Epoch  100  loss  0.07685172546861194 correct 50 time 0.20065808296203613
Epoch  110  loss  0.06599412490781786 correct 50 time 0.20733928680419922
Epoch  120  loss  0.14456477303262125 correct 50 time 0.19671893119812012
Epoch  130  loss  0.07228824538079046 correct 50 time 0.19327878952026367
Epoch  140  loss  0.7586226335370937 correct 50 time 0.19384407997131348
Epoch  150  loss  0.08251264679999547 correct 50 time 0.3702878952026367
Epoch  160  loss  0.002401845716100733 correct 50 time 0.2014920711517334
Epoch  170  loss  0.04948326795859939 correct 50 time 0.20176911354064941
Epoch  180  loss  0.114968244517408 correct 50 time 0.21303105354309082
Epoch  190  loss  0.017124936695195866 correct 50 time 0.19806575775146484
Epoch  200  loss  0.1291958913413667 correct 50 time 0.19643115997314453
Epoch  210  loss  0.02388060494759604 correct 50 time 0.20109176635742188
Epoch  220  loss  0.4042988793568518 correct 50 time 0.20426487922668457
Epoch  230  loss  0.05959669742650808 correct 50 time 0.20039987564086914
Epoch  240  loss  0.033763657230844964 correct 50 time 0.1984708309173584
Epoch  250  loss  0.12792026007372614 correct 50 time 0.21024036407470703
Epoch  260  loss  0.11840649354211029 correct 50 time 0.20865130424499512
Epoch  270  loss  0.15748894341345737 correct 50 time 0.19971752166748047
Epoch  280  loss  0.44908361178457706 correct 50 time 0.19979453086853027
Epoch  290  loss  0.07769578536962497 correct 50 time 0.19786310195922852
Epoch  300  loss  0.3020846457681889 correct 50 time 0.1982860565185547
Epoch  310  loss  0.0798070286085715 correct 50 time 0.2109084129333496
Epoch  320  loss  0.41991630014038905 correct 50 time 0.1970996856689453
Epoch  330  loss  0.0890329430867497 correct 50 time 0.2115318775177002
Epoch  340  loss  0.028326010137385373 correct 50 time 0.21343088150024414
Epoch  350  loss  0.028807221417237246 correct 50 time 0.19782519340515137
Epoch  360  loss  0.24946717947901773 correct 50 time 0.19247913360595703
Epoch  370  loss  0.4223914005682883 correct 50 time 0.19251227378845215
Epoch  380  loss  0.021927729804215212 correct 50 time 0.19242429733276367
Epoch  390  loss  0.006571786964822105 correct 50 time 0.20191431045532227
Epoch  400  loss  0.059072251866149984 correct 50 time 0.20997357368469238
Epoch  410  loss  0.00041338647181229866 correct 50 time 0.19541120529174805
Epoch  420  loss  0.25741736681612387 correct 50 time 0.19763588905334473
Epoch  430  loss  0.24262127311531928 correct 50 time 0.20034289360046387
Epoch  440  loss  0.000375595706539699 correct 50 time 0.2010338306427002
Epoch  450  loss  0.0009723679582381326 correct 50 time 0.20634222030639648
Epoch  460  loss  0.32540003272761947 correct 50 time 0.20800495147705078
Epoch  470  loss  0.031117536813050156 correct 50 time 0.21008634567260742
Epoch  480  loss  0.014411481311765447 correct 50 time 0.21195411682128906
Epoch  490  loss  0.012755036289838432 correct 50 time 0.19680452346801758

Split:
!cd $DIR; PYTHONPATH=/content/$DIR python3.11 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch  0  loss  4.298107157175864 correct 34 time 25.30163288116455
Epoch  10  loss  5.479723088570705 correct 42 time 0.09175324440002441
Epoch  20  loss  4.054513591821425 correct 40 time 0.09463953971862793
Epoch  30  loss  5.388372899194938 correct 43 time 0.19588065147399902
Epoch  40  loss  5.943197861748328 correct 43 time 0.22204351425170898
Epoch  50  loss  2.0590187630609673 correct 44 time 0.09004449844360352
Epoch  60  loss  2.8196388215622 correct 43 time 0.09801983833312988
Epoch  70  loss  2.5211452235671215 correct 46 time 0.09348464012145996
Epoch  80  loss  1.7681894212853246 correct 46 time 0.09426236152648926
Epoch  90  loss  2.5333656325881404 correct 48 time 0.09501433372497559
Epoch  100  loss  2.727803807327815 correct 48 time 0.09451985359191895
Epoch  110  loss  2.195839513325076 correct 50 time 0.10427260398864746
Epoch  120  loss  1.1530343277310802 correct 47 time 0.09334850311279297
Epoch  130  loss  1.0895406049578618 correct 48 time 0.09247899055480957
Epoch  140  loss  0.8693494191066606 correct 50 time 0.09152483940124512
Epoch  150  loss  1.108079965683331 correct 50 time 0.08870434761047363
Epoch  160  loss  1.007631477804777 correct 50 time 0.22035765647888184
Epoch  170  loss  1.0386050096914425 correct 49 time 0.17925333976745605
Epoch  180  loss  0.7550756931942865 correct 49 time 0.09239554405212402
Epoch  190  loss  0.8365696329834659 correct 46 time 0.09524869918823242
Epoch  200  loss  0.7180777626256832 correct 50 time 0.0948479175567627
Epoch  210  loss  1.5797603815962067 correct 50 time 0.09065771102905273
Epoch  220  loss  0.4734817852122462 correct 50 time 0.09652423858642578
Epoch  230  loss  1.0802238491084402 correct 50 time 0.09249472618103027
Epoch  240  loss  1.3091656696169087 correct 47 time 0.0912935733795166
Epoch  250  loss  0.9135701245846087 correct 48 time 0.09396672248840332
Epoch  260  loss  0.9235224294240019 correct 48 time 0.09012007713317871
Epoch  270  loss  1.7077317623015706 correct 50 time 0.08995938301086426
Epoch  280  loss  1.4177730901906218 correct 50 time 0.09540510177612305
Epoch  290  loss  1.4283111319254949 correct 46 time 0.21306633949279785
Epoch  300  loss  1.07193098489494 correct 50 time 0.1859588623046875
Epoch  310  loss  0.6039255579493444 correct 50 time 0.08958864212036133
Epoch  320  loss  1.5867382882715517 correct 50 time 0.09306168556213379
Epoch  330  loss  1.3347303275097622 correct 50 time 0.10241985321044922
Epoch  340  loss  0.6811113906548056 correct 50 time 0.09566020965576172
Epoch  350  loss  1.1087012609499278 correct 50 time 0.09368300437927246
Epoch  360  loss  0.2657721113616709 correct 48 time 0.09390425682067871
Epoch  370  loss  1.112024646553253 correct 48 time 0.0924978256225586
Epoch  380  loss  0.19863492709573066 correct 50 time 0.0929110050201416
Epoch  390  loss  1.0691781472462742 correct 50 time 0.09585881233215332
Epoch  400  loss  0.7225806509725088 correct 48 time 0.11190438270568848
Epoch  410  loss  0.18546764890103254 correct 48 time 0.09426116943359375
Epoch  420  loss  0.919662287172363 correct 48 time 0.22242975234985352
Epoch  430  loss  1.0821449152102005 correct 50 time 0.2021040916442871
Epoch  440  loss  0.09693095386473649 correct 50 time 0.09358716011047363
Epoch  450  loss  0.2658287082537292 correct 50 time 0.09426641464233398
Epoch  460  loss  0.18314981562269514 correct 50 time 0.09221935272216797
Epoch  470  loss  0.9946247449788602 correct 50 time 0.09336209297180176
Epoch  480  loss  0.4436025711351306 correct 48 time 0.09245133399963379
Epoch  490  loss  0.35777676005905357 correct 50 time 0.10616254806518555

Xor:
!cd $DIR; PYTHONPATH=/content/$DIR python3.11 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
Epoch  0  loss  7.714984587649593 correct 29 time 24.60503625869751
Epoch  10  loss  3.906832721372432 correct 39 time 0.09507155418395996
Epoch  20  loss  4.8797380762431715 correct 44 time 0.09920001029968262
Epoch  30  loss  2.321555035107872 correct 44 time 0.10961556434631348
Epoch  40  loss  2.068156803698187 correct 47 time 0.1276566982269287
Epoch  50  loss  3.4726495406126254 correct 46 time 0.18731117248535156
Epoch  60  loss  2.276822936199895 correct 49 time 0.09259772300720215
Epoch  70  loss  2.4578378682405027 correct 47 time 0.09403634071350098
Epoch  80  loss  2.1460919230407702 correct 49 time 0.11293172836303711
Epoch  90  loss  1.8390444032929758 correct 50 time 0.0958864688873291
Epoch  100  loss  0.23454668808354012 correct 50 time 0.093231201171875
Epoch  110  loss  1.595494720620879 correct 50 time 0.10501885414123535
Epoch  120  loss  0.9552354382735608 correct 50 time 0.09140205383300781
Epoch  130  loss  1.0481093748129358 correct 50 time 0.0923459529876709
Epoch  140  loss  0.46614312751672693 correct 50 time 0.09180402755737305
Epoch  150  loss  1.107662641352508 correct 50 time 0.09542107582092285
Epoch  160  loss  0.4861225264687362 correct 50 time 0.09498953819274902
Epoch  170  loss  1.5103042806245768 correct 50 time 0.15674257278442383
Epoch  180  loss  0.6508572719201725 correct 50 time 0.15327191352844238
Epoch  190  loss  1.0737778310061863 correct 50 time 0.09858822822570801
Epoch  200  loss  0.4192083294116308 correct 50 time 0.10002636909484863
Epoch  210  loss  0.21292800634064055 correct 50 time 0.09372448921203613
Epoch  220  loss  0.36920964290540104 correct 50 time 0.09131836891174316
Epoch  230  loss  0.460128524355986 correct 50 time 0.09685420989990234
Epoch  240  loss  0.6657846921026755 correct 50 time 0.10669398307800293
Epoch  250  loss  0.15684683911680153 correct 50 time 0.09785699844360352
Epoch  260  loss  0.10620620265057962 correct 50 time 0.09111618995666504
Epoch  270  loss  0.08093239199381898 correct 50 time 0.09293961524963379
Epoch  280  loss  0.1156212192163377 correct 50 time 0.09287786483764648
Epoch  290  loss  0.036783002954367164 correct 50 time 0.09160137176513672
Epoch  300  loss  0.21966708326049908 correct 50 time 0.11214399337768555
Epoch  310  loss  0.23806190909999417 correct 50 time 0.15506672859191895
Epoch  320  loss  0.14105386112930354 correct 50 time 0.0923004150390625
Epoch  330  loss  0.1517682297718337 correct 50 time 0.10432982444763184
Epoch  340  loss  0.3443906011433613 correct 50 time 0.09268832206726074
Epoch  350  loss  0.12563287799596798 correct 50 time 0.09543561935424805
Epoch  360  loss  0.1933445918345819 correct 50 time 0.09296154975891113
Epoch  370  loss  0.08510773099545889 correct 50 time 0.09405303001403809
Epoch  380  loss  0.5874789579777987 correct 50 time 0.09025025367736816
Epoch  390  loss  0.041079020883295625 correct 50 time 0.09091997146606445
Epoch  400  loss  0.28126624305335296 correct 50 time 0.10056948661804199
Epoch  410  loss  0.18397070615369215 correct 50 time 0.09485864639282227
Epoch  420  loss  0.3096394309109591 correct 50 time 0.09332799911499023
Epoch  430  loss  0.25876241089952395 correct 50 time 0.1530914306640625
Epoch  440  loss  0.21741236802171884 correct 50 time 0.1386430263519287
Epoch  450  loss  0.17258942962113882 correct 50 time 0.09404516220092773
Epoch  460  loss  0.12403664650021547 correct 50 time 0.10138535499572754
Epoch  470  loss  0.25724380780667383 correct 50 time 0.1705923080444336
Epoch  480  loss  0.1617198728640299 correct 50 time 0.09096407890319824
Epoch  490  loss  0.1622962335997073 correct 50 time 0.08983778953552246