from itertools import combinations, product

qubits = 5
weight = 2

strings = []
nontrivial_matrix_elements = list(product(["X", "Y", "Z"], repeat=weight))
for indices in combinations(range(qubits), weight):  # n(n-1)/2 terms
    for elements in nontrivial_matrix_elements:
        pauli_string = []
        for qbit in range(qubits):
            for el_position, i in enumerate(indices):
                if i == qbit:
                    pauli_string.append(elements[el_position])
                    break
            else:
                pauli_string.append("I")
        strings.append("".join(pauli_string))

print(strings)

"XXIII", "XYIII", "XZIII", "YXIII", "YYIII", "YZIII", "ZXIII", "ZYIII", "ZZIII", "XIXII", "XIYII", "XIZII", "YIXII", "YIYII", "YIZII"
