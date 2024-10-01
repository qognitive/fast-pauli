
=====================
Getting Started Guide
=====================

Welcome to Fast-Pauli, a Python / C++ library for optimized operations on Pauli matrices and Pauli strings. In this guide,
we'll introduce some of the important operations to help users get started, as well as some conceptual background. For more details,
see the API documentation **INSERT LINK TO API DOCS**.


Installation
-----------------------
First, we'll guide through the installation. First, you'll need to install the library. You can do this by running:

.. code-block:: bash

    pip install fast-pauli

This will install the library. Next you'll want to build ``fast_pauli`` from source.
Requirements are:

* CMake >= 3.25
* C++ compiler with OpenMP and C++20 support (LLVM recommended)
* Python >= 3.10

A quick start would be:

.. code-block:: bash

    python -m pip install -e ".[dev]"
    pytest -v tests/fast_pauli

Configurable Build operations

.. code-block:: bash

    cmake -B build # + other CMake flags
    cmake --build build --target install --parallel
    ctest --test-dir build

    python -m pip install --no-build-isolation -ve.
    pytest -v tests/fast_pauli # + other pytest flags

After this, the compiled ``_fast_pauli`` python module gets installed into the ``fast_pauli`` directory.

Next, we'll go over some of the important operations, and some of the underlying concepts.

Pauli Matrices
------------------------
In math and physics, a Pauli matrix (often denoted by the greek letter sigma) is any one of the special 2 x 2 complex matrices in the set:

.. math::
    \sigma_0 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
    \sigma_x = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
    \sigma_y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}
    \sigma_z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}

All the pauli matrices share the properties that they are:

1. Hermitian (equal to their own conjugate transpose) :math:`\sigma_i = \sigma_i^\dagger` for all :math:`i \in \{0, x, y, z\}`
2. Involutory (they are their own inverse) :math:`\sigma_i^2 = \sigma_0` for all :math:`i \in \{x, y, z\}`
3. Unitary (their inverse is equal to their conjugate transpose) :math:`\sigma_i^{-1} = \sigma_i^\dagger` for all :math:`i \in \{x, y, z\}`

In ``fast_pauli``, we represent pauli matrices using the ``Pauli`` class. For example, to construct the ``X`` matrix, we can do:

.. code-block:: python

    import fast_pauli as fp

    pauli_0 = fp.Pauli('I')
    pauli_x = fp.Pauli('X')
    pauli_y = fp.Pauli('Y')
    pauli_z = fp.Pauli('Z')

From here, we can also convert our ``Pauli`` object back to a dense numpy array if we'd like:

.. code-block:: python

    pauli_x_np = pauli_x.to_tensor()

PauliString
------------------------

Pauli Strings are tensor-product combinations of Pauli matrices. For example, the following is a valid Pauli string:

.. math::

    P = \sigma_x \otimes \sigma_y \otimes \sigma_z

where :math:`\otimes` denotes the tensor product, and we can more simply denote by

.. math::

    P = XYZ

Other valid Pauli strings include ``III``, ``IXYZ``, ``IZYX``, etc. In general, a Pauli string of length ``N`` is a tensor product of ``N``
Pauli matrices. A ``N``-length Pauli String in dense form is a :math:`2^N \times 2^N` matrix, so ``XYZ`` is a :math:`8 \times 8` matrix.

In ``fast_pauli``, we represent Pauli strings using the ``PauliString`` class. For example, to construct the Pauli string ``X, Y, Z``, we can do:

.. code-block:: python

    P = fp.PauliString('XYZ')

Pauli Strings also support operations like addition, multiplication, equality, and more. For example:

.. code-block:: python

    P1 = fp.PauliString('XYZ')
    P2 = fp.PauliString('YZX')

    # Add two Pauli strings
    P3 = P1 + P2

    # Multiply two Pauli strings
    P4 = P1 @ P2

    # Check if two Pauli strings are equal
    P1 == P2


We can also do more complicated things, like compute the action of a Pauli string :math:`P` on a quantum state :math:`| \psi \rangle`, :math:`P| \psi \rangle`, or
compute the expectation value of a Pauli string with a state :math:`\langle \psi | P | \psi \rangle`:

.. code-block:: python

    # Apply P to a state
    P = fp.PauliString('XY')
    state = np.array([1, 0, 0, 1], dtype=complex)
    state = P.apply(state)

    # Compute the expected value of P with respect to a state
    value = P.expectation_value(state)

We can also convert ``PauliString`` objects back to dense numpy arrays if we'd like:

.. code-block:: python

    P_np = P.to_tensor()

For more details on the ``PauliString`` class, see the **INSERT DOCS PAGE**.

PauliOp
------------------------

The ``PauliOp`` class lets us represent operators that are linear combinations of Pauli strings with complex coefficients. More specifically,
we can represent an arbitrary operator :math:`O` as a sum of Pauli strings :math:`P_i` with complex coefficients :math:`c_i`:

.. math::

    O = \sum_i c_i P_i

In ``fast_pauli``, we can construct ``PauliOp`` objects using the ``PauliOp`` constructor. For example, to construct the ``PauliOp`` object
that represents the operator :math:`O = 0.5 * XYZ + 0.5 * YYZ`, we can do:

.. code-block:: python

    coeffs = np.array([0.5, 0.5], dtype=complex)
    pauli_strings = ['XYZ', 'YYZ']
    O = fp.PauliOp(coeffs, pauli_strings)

Just like with ``PauliString`` objects, we can apply ``PauliOp`` objects to a set of quantum states or compute expectation values, as well as arithmetic
operations and dense matrix conversions.

.. code-block:: python

    # apply O to a set of states
    states = np.random.rand(10, 8) + 1j * np.random.rand(10, 8)
    states = O.apply(states)

    # compute the expected value of O with respect to a state
    value = O.expectation_value(states)

    O_dense = O.to_tensor()

Qiskit Integration
------------------------
``Fast-Pauli`` also has integration with IBM's Qiskit SDK, allowing for easy interfacing with the entire Qiskit ecosystem. For example, we can convert

.. code-block:: python

    # Convert a Fast-Pauli PauliOp to a Qiskit SparsePauliOp object and back
    O = fp.PauliOp([1], ['XYZ'])
    qiskit_op = fp.to_qiskit(O)
    fast_pauli_op = fp.from_qiskit(qiskit_op)

    # Convert a Fast-Pauli PauliString to a Qiskit Pauli object and back
    P = fp.PauliString('XYZ')
    qiskit_pauli = fp.to_qiskit(P)
    fp_pauliString = fp.from_qiskit(qiskit_pauli)

For more details on the ``PauliOp`` class, see the **INSERT DOCS PAGE**.

