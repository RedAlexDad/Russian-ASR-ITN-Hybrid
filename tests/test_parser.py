import pytest
from src.parser import parse_number_group


def test_sum_hundreds_tens():
    assert parse_number_group([(200, 3, False, False), (50, 2, False, False)]) == ['250']


def test_enumeration_same_magnitude():
    assert parse_number_group([(200, 3, False, False), (300, 3, False, False)]) == ['200', '300']


def test_sum_with_thousand_multiplier():
    assert parse_number_group([(2, 0, False, False), (1000, 4, True, False),
                                (800, 3, False, False), (40, 2, False, False),
                                (3, 0, False, False)]) == ['2843']


def test_two_multiplier_blocks():
    # Два блока одного ранга с числом между — разные числа
    # "семьдесят миллионов два миллиона" → 70000000 2000000
    assert parse_number_group([(70, 2, False, False), (1000000, 5, True, False),
                                (2, 0, False, False), (1000000, 5, True, False)]) == ['70000000', '2000000']


def test_standalone_thousand():
    assert parse_number_group([(1000, 4, True, False)]) == ['1000']


def test_zero():
    assert parse_number_group([(0, 0, False, False)]) == ['0']


def test_zero_and_seven():
    assert parse_number_group([(0, 0, False, False), (7, 0, False, False)]) == ['0', '7']


def test_ordinal_in_compound():
    assert parse_number_group([(200, 3, False, False), (80, 1, False, False),
                                (5, 0, False, True)]) == ['285']


def test_empty():
    assert parse_number_group([]) == []
