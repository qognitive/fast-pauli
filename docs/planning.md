# General


## Notation

- Pauli Matrix $\sigma_i \in \{ I,X,Y,Z \}$
- Pauli String $\mathcal{\hat{P}} = \bigotimes_i \sigma_i$
- Sum of weighted Pauli strings (currently called `PauliOp`) $A_k = \sum_i h_i \mathcal{\hat{P_i}}$
- Sum of summed weighted Pauli strings (currently called `SummedPauliOp`) $B = \sum_k \sum_i h_{ik} \mathcal{\hat{P_i}}$

# List of Operations

Here's a terse list of the type of operations we want to support in `fast_pauli` (this list will grow over time):

1. Pauli String to sparse matrix (Pauli Composer)
2. $\mathcal{\hat{P}} \ket{\psi}$
3. $\mathcal{\hat{P}} \ket{\psi_t}$
4. $\big( \sum_i h_i \mathcal{\hat{P}}_i \big) \ket{\psi_t}$
5. $\big(\sum_k \sum_i h_{ik} \mathcal{\hat{P}}_i \big) \ket{\psi_t}$
6. $\big(\sum_k x_{tk} \sum_i h_{ik} \mathcal{\hat{P}}_i \big) \ket{\psi_t}$
7. $\bigg(\sum_k \big( \sum_i h_{ik} \mathcal{\hat{P}}_i \big)^2 \bigg) \ket{\psi_t}$
8. Calculate $\bra{\psi_t} \{ \mathcal{\hat{P_i}}, \hat{A_k} \} \ket{\psi}$ and $\bra{\psi} \mathcal{\hat{P_i}} \ket{\psi}$
