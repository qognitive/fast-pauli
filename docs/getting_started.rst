
=====================
Getting Started
=====================

Welcome to :code:`fast-pauli` from `Qognitive <https://www.qognitive.io/>`_, an open-source Python / C++ library for optimized operations on Pauli matrices and Pauli strings
based on `PauliComposer <https://arxiv.org/abs/2301.00560>`_.
In this guide, we'll introduce some of the important operations to help users get started as well as some conceptual background on Pauli matrices and Pauli strings.

For more details, see the :doc:`python_api` or :doc:`cpp_api` documentation.

For tips on installing the library, check out the guide: :doc:`index`.

Pauli Matrices
------------------------

For a more in-depth overview of Pauli matrices, see `here <https://en.wikipedia.org/wiki/Pauli_matrices>`_.

In math and physics, a `Pauli matrix <https://en.wikipedia.org/wiki/Pauli_matrices>`_, named after the physicist Wolfgang Pauli, is any one of the special 2 x 2 complex matrices in the set (often denoted by the greek letter :math:`\sigma`) :

.. math::

    \sigma_x = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
    \sigma_y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}
    \sigma_z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}

All the pauli matrices share the properties that they are:

1. Hermitian (equal to their own conjugate transpose) :math:`\sigma_i = \sigma_i^\dagger` for all :math:`i \in \{x, y, z\}`
2. Involutory (they are their own inverse) :math:`\sigma_i^2 = \sigma_0` for all :math:`i \in \{x, y, z\}`
3. Unitary (their inverse is equal to their conjugate transpose) :math:`\sigma_i^{-1} = \sigma_i^\dagger` for all :math:`i \in \{x, y, z\}`

with the identity matrix :math:`\sigma_0` or the :math:`2 \times 2` Identity matrix :math:`I` being the trivial case.

In ``fast_pauli``, we represent pauli matrices using the ``Pauli`` class. For example, to represent the Pauli matrices, we can do:

.. code-block:: python

    import fast_pauli as fp

    pauli_0 = fp.Pauli('I')
    pauli_x = fp.Pauli('X')
    pauli_y = fp.Pauli('Y')
    pauli_z = fp.Pauli('Z')

    str(pauli_0)  # returns "I"

We can also multiply two ``Pauli`` objects together to get a ``Pauli`` object representing the matrix product of the two pauli matrices.
The result includes a phase factor because the product of two Pauli matrices is not another Pauli matrix.

For example, we can compute the resulting ``Pauli`` object from multiplying :math:`\sigma_x` and :math:`\sigma_y` as follows:

.. code-block:: python

    # phase = i, new_pauli = fp.Pauli('Z')
    phase, new_pauli = pauli_x @ pauli_y

From here, we can also convert our ``Pauli`` object back to a dense numpy array if we'd like:

.. code-block:: python

    pauli_x_np = pauli_x.to_tensor()

Pauli Strings
------------------------

Pauli strings are tensor-product combinations of Pauli matrices. For example, the following is a valid Pauli string:

.. math::

    \mathcal{\hat{P}} = \sigma_x \otimes \sigma_y \otimes \sigma_z

where :math:`\otimes` denotes the tensor or `Kronecker <https://en.wikipedia.org/wiki/Kronecker_product>`_ product, and we can more simply denote by

.. math::

    \mathcal{\hat{P}} = XYZ

A Pauli string of length ``N`` is a tensor product of ``N``
Pauli matrices. A ``N``-length Pauli String is a :math:`2^N \times 2^N` matrix, so ``XYZ`` is a :math:`8 \times 8` matrix.

In ``fast_pauli``, we represent Pauli strings using the ``PauliString`` class. For example, to construct the Pauli string ``X, Y, Z``, we can do:

.. code-block:: python

    P = fp.PauliString('XYZ')

Pauli Strings also support operations like addition, multiplication, and more. For example:

.. code-block:: python

    P1 = fp.PauliString('XYZ')
    P2 = fp.PauliString('YZX')

    # Get dim and n_qubits properties
    # dim = 8, n_qubits = 3
    P1.dim
    P1.n_qubits

    # Multiply two Pauli strings.
    phase, new_string = P1 @ P2


We can also do more complicated things, like compute the action of a Pauli string :math:`\mathcal{\hat{P}}` on a vector :math:`| \psi \rangle`, :math:`\mathcal{\hat{P}}| \psi \rangle`, or
compute the expectation value of a Pauli string with a state :math:`\langle \psi | \mathcal{\hat{P}} | \psi \rangle`. As a side note, in this guide we will use state and vector interchangeably:

.. code-block:: python

    # Apply P to a state
    P = fp.PauliString('XY')
    state = np.array([1, 0, 0, 1], dtype=complex)
    new_state = P.apply(state)

    # Compute the expected value of P with respect to a state or a batch of states
    value = P.expectation_value(state)

    states = np.random.randn(8, 8) + 1j * np.random.randn(8, 8)
    values = P.expectation_value(states)

We can also convert ``PauliString`` objects back to dense numpy arrays if we'd like, or extract their string representation:

.. code-block:: python

    P = fp.PauliString('XYZ')
    P_np = P.to_tensor()

    P_str = str(P) # Returns "XYZ"

For more details on the ``PauliString`` class, see the :doc:`python_api` or :doc:`cpp_api` documentation.

Pauli Operators
------------------------

The ``PauliOp`` class lets us represent operators that are linear combinations of Pauli strings with complex coefficients.
In physics, an operator is represented by a matrix in a given basis.
For example, we can represent any arbitrary operator :math:`A` as a sum of Pauli strings :math:`P_i` with complex coefficients :math:`c_i`:

.. math::

    A = \sum_i c_i P_i

In ``fast_pauli``, we can construct ``PauliOp`` objects using the ``PauliOp`` constructor. For example, to construct the ``PauliOp`` object
that represents the operator :math:`A = 0.5 * XYZ + 0.5 * YYZ`, we can do:

.. code-block:: python

    coeffs = np.array([0.5, 0.5], dtype=complex)  # represent c_i in the sum above
    pauli_strings = ['XYZ', 'YYZ']  # represent P_i in the sum above
    A = fp.PauliOp(coeffs, pauli_strings)

    # Get the number of qubits the operator acts on,
    # dimension, number of pauli strings
    # n_qubits = 3, dim = 8, n_pauli_strings = 2
    A.n_qubits
    A.dim
    A.n_pauli_strings

Just like with ``PauliString`` objects, we can apply ``PauliOp`` objects to a set of vectors, or compute expectation values, as well as arithmetic
operations. Just like with ``PauliString`` objects, we can also convert ``PauliOp`` objects back to dense numpy arrays if we'd like
or get their string representation, in this case a list of strings:

.. code-block:: python

    coeffs = np.array([0.5, 0.5], dtype=complex)
    pauli_strings = ['XYZ', 'YYZ']
    A = fp.PauliOp(coeffs, pauli_strings)

    # Adding two Pauli strings returns a PauliOp.
    # The returned object is a PauliOp because
    # the sum is a linear combination of Pauli strings
    P1 = fp.PauliString('XYZ')
    P2 = fp.PauliString('YZX')
    O = P1 + P2

    # PauliOp supports addition, subtraction, multiplication,
    # scaling, as well as have PauliString objects
    # as the second operand. All valid operations:
    A1 = 0.5 * A
    A2 = A + A1
    A3 = A1 @ A2
    s = fp.PauliString('XYZ')
    A4 = A1 + s

    # Apply A to a state / vector or set of states
    states = np.random.rand(10, 8) + 1j * np.random.rand(10, 8)
    new_states = A.apply(states)

    # Compute the expectation value of A with respect to a state
    values = A.expectation_value(states)

    # Get dense matrix representation of A
    A_dense = A.to_tensor()

    # ['XYZ', 'YYZ']
    A_str = A.pauli_strings_as_str

Qiskit Integration
------------------------
``Fast-Pauli`` also has integration with `IBM's Qiskit SDK <https://www.ibm.com/quantum/qiskit>`_, allowing for easy interfacing with certain Qiskit objects. For example, we can convert
between ``PauliOp`` objects and ``SparsePauliOp`` objects from Qiskit:

.. code-block:: python

    # Convert a Fast-Pauli PauliOp to a Qiskit SparsePauliOp object and back
    O = fp.PauliOp([1], ['XYZ'])
    qiskit_op = fp.to_qiskit(O)
    fast_pauli_op = fp.from_qiskit(qiskit_op)

    # Convert a Fast-Pauli PauliString to a Qiskit Pauli object
    P = fp.PauliString('XYZ')
    qiskit_pauli = fp.to_qiskit(P)

For more details on Qiskit conversions, see the :doc:`python_api` or :doc:`cpp_api` documentation.

