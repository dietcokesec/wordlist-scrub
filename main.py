import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Union

from rich import print
from rich.progress import track
from typer import Option, Typer

app = Typer()


# Symbols are printable characters that aren't alphanumeric or whitespace.
SYMBOLS = (
    set(string.printable)
    - set(string.ascii_letters)
    - set(string.digits)
    - set(string.whitespace)
)


class ValueType(Enum):
    UPPER = "upper"
    LOWER = "lower"
    DIGIT = "digit"
    SYMBOL = "symbol"


class EvaluationType(Enum):
    COUNT = "count"
    THRESHOLD = "threshold"


@dataclass
class CountCriteria:
    type: EvaluationType
    min_count: int
    value_type: ValueType


@dataclass
class ThresholdCriteria:
    type: EvaluationType
    min_threshold: int
    # Length is implied for threshold criteria

    def __post_init__(self):
        # Ensure threshold criteria is always for length
        pass


CriteriaType = Union[CountCriteria, ThresholdCriteria]


class Evaluation(ABC):
    def __init__(self, criteria: CriteriaType):
        self.criteria = criteria

    @abstractmethod
    def eval(self, s: str) -> bool: ...


class UppercaseEvaluation(Evaluation):
    def __init__(self, criteria: CountCriteria):
        super().__init__(criteria)

    def eval(self, s: str) -> bool:
        count = sum(1 for c in s if c.isupper())
        return count >= self.criteria.min_count  # type: ignore


class LowercaseEvaluation(Evaluation):
    def __init__(self, criteria: CountCriteria):
        super().__init__(criteria)

    def eval(self, s: str) -> bool:
        count = sum(1 for c in s if c.islower())
        return count >= self.criteria.min_count  # type: ignore


class DigitEvaluation(Evaluation):
    def __init__(self, criteria: CountCriteria):
        super().__init__(criteria)

    def eval(self, s: str) -> bool:
        count = sum(1 for c in s if c.isdigit())
        return count >= self.criteria.min_count  # type: ignore


class SymbolEvaluation(Evaluation):
    def __init__(self, criteria: CountCriteria):
        super().__init__(criteria)

    def eval(self, s: str) -> bool:
        count = sum(1 for c in s if c in SYMBOLS)
        return count >= self.criteria.min_count  # type: ignore


class LengthEvaluation(Evaluation):
    def __init__(self, criteria: ThresholdCriteria):
        super().__init__(criteria)

    def eval(self, s: str) -> bool:
        return len(s) >= self.criteria.min_threshold  # type: ignore


class EvaluationFactory:
    """Factory to create appropriate evaluation instances"""

    _character_evaluations = {
        ValueType.UPPER: UppercaseEvaluation,
        ValueType.LOWER: LowercaseEvaluation,
        ValueType.DIGIT: DigitEvaluation,
        ValueType.SYMBOL: SymbolEvaluation,
    }

    @classmethod
    def create(cls, criteria: CriteriaType) -> Evaluation:
        if criteria.type == EvaluationType.COUNT:
            evaluation_class = cls._character_evaluations.get(
                criteria.value_type  # type: ignore
            )
            if not evaluation_class:
                raise ValueError(
                    f"Unknown value type for count criteria: {criteria.value_type}"  # type: ignore
                )
            return evaluation_class(criteria)
        elif criteria.type == EvaluationType.THRESHOLD:
            return LengthEvaluation(criteria)  # type: ignore
        else:
            raise ValueError(f"Unknown criteria type: {criteria.type}")


class WordlistFilter:
    """Main class to filter wordlists based on multiple criteria"""

    def __init__(self, criteria_list: List[CriteriaType]):
        self.evaluations = [
            EvaluationFactory.create(criteria) for criteria in criteria_list
        ]

    def passes_criteria(self, password: str) -> bool:
        """Check if a password passes ALL criteria"""
        return all(
            evaluation.eval(password) for evaluation in self.evaluations
        )

    def filter_wordlist(self, wordlist: List[str]) -> List[str]:
        """Filter wordlist to only include passwords that meet all criteria"""
        return [
            password for password in wordlist if self.passes_criteria(password)
        ]

    def filter_file(self, input_file: Path, output_file: Path) -> int:
        """Filter a wordlist file and save results to output file"""
        kept_count = 0

        if not input_file.exists():
            raise ValueError(
                f"Input file '{input_file.absolute()}' does not exist."
            )

        if not input_file.is_file():
            raise ValueError(
                f"Input file '{input_file.absolute()}' is not a file."
            )

        with (
            open(input_file, "r", encoding="utf-8", errors="ignore") as infile,
            open(output_file, "w+", encoding="utf-8") as outfile,
        ):
            for line in track(infile):
                password = line.strip()
                if self.passes_criteria(password):
                    outfile.write(password + "\n")
                    kept_count += 1

        return kept_count


# Convenience functions for creating criteria
def min_uppercase(count: int) -> CountCriteria:
    return CountCriteria(
        type=EvaluationType.COUNT, min_count=count, value_type=ValueType.UPPER
    )


def min_lowercase(count: int) -> CountCriteria:
    return CountCriteria(
        type=EvaluationType.COUNT, min_count=count, value_type=ValueType.LOWER
    )


def min_digits(count: int) -> CountCriteria:
    return CountCriteria(
        type=EvaluationType.COUNT, min_count=count, value_type=ValueType.DIGIT
    )


def min_symbols(count: int) -> CountCriteria:
    return CountCriteria(
        type=EvaluationType.COUNT, min_count=count, value_type=ValueType.SYMBOL
    )


def min_length(length: int) -> ThresholdCriteria:
    return ThresholdCriteria(
        type=EvaluationType.THRESHOLD, min_threshold=length
    )


@app.command()
def main(
    inp: Path,
    outp: Path,
    min_uppercase_vals: int | None = Option(
        default=None, help="The number of uppercase values to filter for."
    ),
    min_lowercase_vals: int | None = Option(
        default=None, help="The number of lowercase values to filter for."
    ),
    min_digit_vals: int | None = Option(
        default=None, help="The number of digits to filter for."
    ),
    min_symbol_vals: int | None = Option(
        default=None, help=f"The number of symbols to filter for ({SYMBOLS})."
    ),
    min_val_length: int | None = Option(
        default=None, help="The minimum length of the string value."
    ),
):
    criteria: List[CriteriaType] = list(
        filter(
            bool,
            [
                min_uppercase(min_uppercase_vals)
                if min_uppercase_vals
                else None,
                min_lowercase(min_lowercase_vals)
                if min_lowercase_vals
                else None,
                min_digits(min_digit_vals) if min_digit_vals else None,
                min_symbols(min_symbol_vals) if min_symbol_vals else None,
                min_length(min_val_length) if min_val_length else None,
            ],
        )
    )  # type: ignore

    if not any(criteria):
        print("[red]We need at least one criteria.")
        exit(1)

    # Create filter
    filter_instance = WordlistFilter(criteria)

    # Filter
    kept = filter_instance.filter_file(inp, outp)

    print(f"[green]Kept {kept} passwords.")


app()
