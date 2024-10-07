Benchmarks
==========

Here at `Qognitive <https://www.qognitive.io/>`_, we use Pauli Strings and Pauli Operators throughout our codebase.
Some of our most critical performance bottlenecks involve applying these operators to a state or batch of states.
These are only a few of the functions we've optimized in :code:`fast-pauli`, check out our :doc:`Python API <python_api>` and :doc:`C++ API <cpp_api>`.
All benchmark figures are interactive and we encourage you to explore them!

Below are several benchmarks comparing :code:`fast-pauli` to :code:`qiskit`.
All benchmarks were run on a single machine with the following specifications:

.. list-table::
    :header-rows: 0
    :widths: 40 60

    * - CPU
      - 13th Gen Intel(R) Core(TM) i9-13950HX
    * - RAM
      - 64GB
    * - Threads
      - 32
    * - OS
      - Ubuntu 22.04.4 LTS
    * - Architecture
      - x86_64
    * - Compiler (for :code:`fast-pauli`)
      - LLVM 18.1.8
    * - Python
      - 3.12.7


Pauli String Applied to a State
-------------------------------

Starting simply, we benchmarked applying a single Pauli String (:math:`\mathcal{\hat{P}}`) to a single state (:math:`\ket{\psi}`), which is equivalent to the following expression:

.. math::
    :label: pauli_string_apply

    \mathcal{\hat{P}} \ket{\psi}

.. raw:: html
    :file: benchmark_results/figs/qiskit_pauli_string_apply.html

We saw that the sparse representation of the Pauli String operator when applied to the state is significantly faster than the representation of the Pauli String operator used by Qiskit.
For most operator sizes, we saw several orders of magnitude in performance improvement.

.. note::
    All datapoints in our benchmarks have error bars indicating the standard deviation of the mean, but for most points the error bars are too small to see.

Pauli Operator Applied to a State
---------------------------------

Next we benchmarked applying a Pauli Operator (a linear combination of Pauli Strings) to a single state:

.. math::
    :label: pauli_op_apply

    \big( \sum_i c_i \mathcal{\hat{P_i}} \big) \ket{\psi}

.. raw:: html
    :file: benchmark_results/figs/qiskit_pauli_op_apply.html

Again, we saw significant performance improvements for the same reasons stated above and are often an order of magnitude faster than :code:`qiskit`.
:math:`N_{\text{pauli strings}}` is the number of Pauli Strings in the Pauli Operator, i.e. the number of terms in the linear combination shown in :eq:`pauli_op_apply`.
Note that :code:`fast-pauli` performs better when the Pauli Operator is more sparse.


Expectation Value of a Pauli Operator
-------------------------------------------------------------------

Finally, we benchmarked the expectation value of a Pauli Operator applied to a **batch** of states:

.. math::
    :label: pauli_op_expectation_value

    \bra{\psi_t} \sum_i c_i \mathcal{\hat{P_i}} \ket{\psi_t}

.. raw:: html
    :file: benchmark_results/figs/qiskit_pauli_op_expectation_value_batch.html

Similar to the previous benchmarks, we saw significant performance improvements for :code:`fast-pauli` compared to :code:`qiskit`.
In general, we tend to perform better when applying to a larger batch of states, but we point out that our advantage compared to :code:`qiskit` narrows as the number of qubits increases.
With that said, we're still more than 2x faster for these larger operators!