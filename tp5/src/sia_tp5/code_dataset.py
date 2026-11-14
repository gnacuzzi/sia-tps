"""Dataset sintetico de funciones aritmeticas Python de longitud fija."""

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


OPERATORS = ("+", "-", "*")
CONSTANTS = tuple(range(100))
SOURCE_LENGTH = 26
VOCABULARY = tuple(
    " ()*+-0123456789:defnrtux"
)
VOCABULARY_SIZE = len(VOCABULARY)
CHARACTER_TO_INDEX = {
    character: index for index, character in enumerate(VOCABULARY)
}


@dataclass(frozen=True)
class CodeDataset:
    sources: tuple
    patterns: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray

    @property
    def train_patterns(self):
        return self.patterns[self.train_indices]

    @property
    def test_patterns(self):
        return self.patterns[self.test_indices]

    @property
    def validation_patterns(self):
        return self.patterns[self.validation_indices]

    @property
    def train_sources(self):
        return tuple(self.sources[index] for index in self.train_indices)

    @property
    def test_sources(self):
        return tuple(self.sources[index] for index in self.test_indices)

    @property
    def validation_sources(self):
        return tuple(self.sources[index] for index in self.validation_indices)


def generate_function_sources():
    """Enumerar el universo de 90.000 funciones de la gramatica."""
    return tuple(
        f"def f(x): return (x{left_operator}{left_constant:2d})"
        f"{right_operator}{right_constant:2d}"
        for left_operator in OPERATORS
        for left_constant in CONSTANTS
        for right_operator in OPERATORS
        for right_constant in CONSTANTS
    )


def encode_sources(sources):
    """Codificar cada posicion como un one-hot sobre el vocabulario."""
    sources = tuple(sources)
    if not sources:
        raise ValueError("sources no puede estar vacio")
    if any(len(source) != SOURCE_LENGTH for source in sources):
        raise ValueError(f"cada fuente debe tener {SOURCE_LENGTH} caracteres")
    try:
        indices = np.asarray([
            [CHARACTER_TO_INDEX[character] for character in source]
            for source in sources
        ])
    except KeyError as error:
        raise ValueError(f"caracter fuera del vocabulario: {error.args[0]!r}")
    one_hot = np.eye(VOCABULARY_SIZE, dtype=np.float64)[indices]
    return one_hot.reshape(len(sources), SOURCE_LENGTH * VOCABULARY_SIZE)


def decode_patterns(patterns):
    """Elegir por argmax un caracter del vocabulario en cada posicion."""
    values = np.asarray(patterns, dtype=np.float64)
    expected_width = SOURCE_LENGTH * VOCABULARY_SIZE
    if (values.ndim != 2 or len(values) == 0
            or values.shape[1] != expected_width
            or not np.isfinite(values).all()):
        raise ValueError(
            f"patterns debe tener forma (muestras, {expected_width})"
        )
    grouped = values.reshape(len(values), SOURCE_LENGTH, VOCABULARY_SIZE)
    indices = np.argmax(grouped, axis=2)
    return tuple(
        "".join(VOCABULARY[index] for index in row) for row in indices
    )


def compiles(source):
    """Comprobar sintaxis y generacion de bytecode sin ejecutar la fuente."""
    try:
        compile(source, "<generated>", "exec")
    except (SyntaxError, ValueError, TypeError, OverflowError):
        return False
    return True


def belongs_to_grammar(source):
    """Verificar la estructura AST exacta de la familia elegida."""
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError, TypeError):
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False
    function = tree.body[0]
    arguments = function.args
    if (function.name != "f" or function.decorator_list
            or len(arguments.args) != 1 or arguments.args[0].arg != "x"
            or arguments.posonlyargs or arguments.kwonlyargs
            or arguments.vararg is not None or arguments.kwarg is not None
            or arguments.defaults or arguments.kw_defaults != []
            or len(function.body) != 1
            or not isinstance(function.body[0], ast.Return)):
        return False
    outer = function.body[0].value
    if not isinstance(outer, ast.BinOp) or not isinstance(
            outer.op, (ast.Add, ast.Sub, ast.Mult)):
        return False
    inner = outer.left
    if (not isinstance(inner, ast.BinOp)
            or not isinstance(inner.op, (ast.Add, ast.Sub, ast.Mult))
            or not isinstance(inner.left, ast.Name)
            or inner.left.id != "x"):
        return False

    def is_allowed_constant(node):
        return (isinstance(node, ast.Constant)
                and isinstance(node.value, int)
                and not isinstance(node.value, bool)
                and 0 <= node.value <= 99)

    return (is_allowed_constant(inner.right)
            and is_allowed_constant(outer.right))


def _operator_symbol(operator):
    return {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}[type(operator)]


def function_parameters(source):
    """Extraer operadores y constantes de una fuente perteneciente a la familia."""
    if not belongs_to_grammar(source):
        raise ValueError("source no pertenece a la gramatica")
    outer = ast.parse(source).body[0].body[0].value
    inner = outer.left
    return (
        _operator_symbol(inner.op), int(inner.right.value),
        _operator_symbol(outer.op), int(outer.right.value),
    )


def semantic_signature(source):
    """Devolver ``(a, b)`` para la funcion exacta ``f(x) = a*x + b``."""
    left_operator, left_constant, right_operator, right_constant = (
        function_parameters(source)
    )
    coefficient, intercept = 1, 0
    for operator, constant in (
            (left_operator, left_constant),
            (right_operator, right_constant)):
        if operator == "+":
            intercept += constant
        elif operator == "-":
            intercept -= constant
        else:
            coefficient *= constant
            intercept *= constant
    return coefficient, intercept


def load_code_dataset(*, split_seed=17, train_per_operator_pair=200,
                      validation_per_operator_pair=50,
                      test_per_operator_pair=50):
    """Muestrear train/validation/test balanceados por par de operadores."""
    counts = (
        train_per_operator_pair,
        validation_per_operator_pair,
        test_per_operator_pair,
    )
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 1
           for count in counts):
        raise ValueError("los tamanos por par deben ser enteros positivos")
    if sum(counts) > len(CONSTANTS) ** 2:
        raise ValueError("los splits superan las combinaciones de constantes")
    generator = np.random.default_rng(split_seed)
    train_sources = []
    validation_sources = []
    test_sources = []
    train_end = train_per_operator_pair
    validation_end = train_end + validation_per_operator_pair
    test_end = validation_end + test_per_operator_pair
    for left_operator in OPERATORS:
        for right_operator in OPERATORS:
            choices = generator.permutation(len(CONSTANTS) ** 2)[:test_end]
            rendered = [
                f"def f(x): return (x{left_operator}{int(choice // 100):2d})"
                f"{right_operator}{int(choice % 100):2d}"
                for choice in choices
            ]
            train_sources.extend(rendered[:train_end])
            validation_sources.extend(rendered[train_end:validation_end])
            test_sources.extend(rendered[validation_end:test_end])
    sources = tuple(train_sources + validation_sources + test_sources)
    patterns = encode_sources(sources)
    train_size = len(train_sources)
    validation_size = len(validation_sources)
    train_indices = np.arange(train_size)
    validation_indices = np.arange(
        train_size, train_size + validation_size
    )
    test_indices = np.arange(train_size + validation_size, len(sources))
    return CodeDataset(
        sources=sources,
        patterns=patterns,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def export_dataset(path, dataset):
    """Versionar fuentes, campos y particion en un CSV legible."""
    train = set(dataset.train_indices.tolist())
    validation = set(dataset.validation_indices.tolist())
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "index", "split", "source", "left_operator", "left_constant",
            "right_operator", "right_constant", "semantic_a", "semantic_b",
        ])
        writer.writeheader()
        for index, source in enumerate(dataset.sources):
            left_operator, left_constant, right_operator, right_constant = (
                function_parameters(source)
            )
            coefficient, intercept = semantic_signature(source)
            writer.writerow({
                "index": index,
                "split": (
                    "train" if index in train
                    else "validation" if index in validation
                    else "test"
                ),
                "source": source,
                "left_operator": left_operator,
                "left_constant": left_constant,
                "right_operator": right_operator,
                "right_constant": right_constant,
                "semantic_a": coefficient,
                "semantic_b": intercept,
            })


def main():
    dataset = load_code_dataset()
    export_dataset(Path("data/python_functions.csv"), dataset)
    print({
        "samples": len(dataset.sources),
        "train": len(dataset.train_indices),
        "validation": len(dataset.validation_indices),
        "test": len(dataset.test_indices),
        "features": dataset.patterns.shape[1],
    })


if __name__ == "__main__":
    main()
