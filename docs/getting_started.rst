
=====================
Getting Started
=====================

Welcome to Fast-Pauli from `Qognitive <https://www.qognitive.io/>`_, an open-source Python / C++ library for optimized operations on Pauli matrices and Pauli strings
based on `PauliComposer <https://arxiv.org/abs/2301.00560>`_. In this guide,
we'll introduce some of the important operations to help users get started. For more details,
see the API documentation.

For tips on installing the library, check out the guide: :doc:`install`.

Pauli Matrices
------------------------

For a conceptual overview of Pauli matrices, see `here <https://en.wikipedia.org/wiki/Pauli_matrices>`_.

In ``fast_pauli``, we represent pauli matrices using the ``Pauli`` class. For example, to represent the Pauli matrices, we can do:

.. code-block:: python

    import fast_pauli as fp

    pauli_0 = fp.Pauli('I')
    pauli_x = fp.Pauli('X')
    pauli_y = fp.Pauli('Y')
    pauli_z = fp.Pauli('Z')

We can also multiply two ``Pauli`` objects together to get a ``Pauli`` object representing the tensor product of the two pauli matrices.

.. code-block:: python

    phase, new_pauli = pauli_x @ pauli_y
    # returns "I"
    str(pauli_0)

From here, we can also convert our ``Pauli`` object back to a dense numpy array if we'd like:

.. code-block:: python

    pauli_x_np = pauli_x.to_tensor()

Pauli Strings
------------------------

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

    # Add two Pauli strings. Return type is a PauliOp because
    # the product is not a Pauli string
    P3 = P1 + P2

    # Multiply two Pauli strings.
    phase, new_string = P1 @ P2


We can also do more complicated things, like compute the action of a Pauli string :math:`\mathcal{\hat{P}}` on a quantum state :math:`| \psi \rangle`, :math:`\mathcal{\hat{P}}| \psi \rangle`, or
compute the expectation value of a Pauli string with a state :math:`\langle \psi | \mathcal{\hat{P}} | \psi \rangle`:

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
    # Returns "XYZ"
    P_str = str(P)

For more details on the ``PauliString`` class, see the Python or C++ API documentation.

Pauli Operators
------------------------

The ``PauliOp`` class lets us represent operators that are linear combinations of Pauli strings with complex coefficients. More specifically,
we can represent an arbitrary operator :math:`A` as a sum of Pauli strings :math:`P_i` with complex coefficients :math:`c_i`:

.. math::

    A = \sum_i c_i P_i

In ``fast_pauli``, we can construct ``PauliOp`` objects using the ``PauliOp`` constructor. For example, to construct the ``PauliOp`` object
that represents the operator :math:`A = 0.5 * XYZ + 0.5 * YYZ`, we can do:

.. code-block:: python

    coeffs = np.array([0.5, 0.5], dtype=complex)
    pauli_strings = ['XYZ', 'YYZ']
    A = fp.PauliOp(coeffs, pauli_strings)

    # Get the number of qubits the operator acts on,
    # dimension, number of pauli strings
    # n_qubits = 3, dim = 8, n_pauli_strings = 2
    A.n_qubits
    A.dim
    A.n_pauli_strings

Just like with ``PauliString`` objects, we can apply ``PauliOp`` objects to a set of quantum states or compute expectation values, as well as arithmetic
operations and dense matrix conversions. Just like with ``PauliString`` objects, we can also convert ``PauliOp`` objects back to dense numpy arrays if we'd like
or get their string representation, in this case a list of strings:

.. code-block:: python

    coeffs = np.array([0.5, 0.5], dtype=complex)
    pauli_strings = ['XYZ', 'YYZ']
    A = fp.PauliOp(coeffs, pauli_strings)

    # PauliOp supports addition, subtraction, multiplication,
    # scaling, as well as have PauliString objects
    # as the second operand. All valid operations:
    A1 = 0.5 * A
    A2 = A + A1
    A3 = A1 @ A2
    s = fp.PauliString('XYZ')
    A4 = A1 + s

    # Apply A to a state or set of states
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

For more details on Qiskit conversions, see the Python or C++ API documentation.

