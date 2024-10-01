# Contributing to Fast Pauli

We'd love to accept your contributions to this project. There are just a few guidelines you need to follow.

1. Please read and follow the [Code of Conduct](CODE_OF_CONDUCT.md)
2. Search [existing issues](https://github.com/qognitive/fast-pauli/issues) for what you are interested in.
3. If you don't find anything, [open a new issue](https://github.com/qognitive/fast-pauli/issues/new/choose).
4. If you have a fix or new feature that you would like to contribute, please [open a new pull request](https://github.com/qognitive/fast-pauli/compare). Please see the [Developer Setup](#developer-setup) section details about setting up your local workflow.
5. If you're adding new functionality, please also add tests and update the documentation.


## Developer Setup

We use [pre-commit](https://pre-commit.com/) to facilitate code quality checks before commits are merged into the repository and ensure that the code is checked during CI/CD.

```bash
# From root project dir
pre-commit install # installs the checks as pre-commit hooks
python -m pip install -e ".[dev]"
```

### Design Choices

The C++ portion of this library relies heavily on spans and views.
These lightweight accessors are helpful and performant, but can lead to dangling spans or accessing bad memory if used improperly.
Developers should familiarize themselves with these dangers by reviewing [this post](https://hackingcpp.com/cpp/std/span.html).

