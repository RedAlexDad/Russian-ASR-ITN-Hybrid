"""Tests for token_classifier + sequence_parser."""

from src.token_classifier import TokenClass, classify
from src.sequence_parser import parse_sequence


# ── Helper ──

def TC(v, m, s):
    return TokenClass(v, m, s, 'test')


# ── Classifier tests ──


def test_classify_unit():
    tc = classify("пять")
    assert tc and tc[0].value == 5 and tc[0].mag == 0


def test_classify_teen():
    tc = classify("четырнадцать")
    assert tc and tc[0].value == 14 and tc[0].mag == 1


def test_classify_ten():
    tc = classify("восемьдесят")
    assert tc and tc[0].value == 80


def test_classify_hundred():
    tc = classify("триста")
    assert tc and tc[0].value == 300


def test_classify_ordinal():
    tc = classify("двадцатое")
    assert tc and tc[0].value == 20 and tc[0].subtype == 'ordinal'


def test_classify_non_numeral():
    assert classify("компьютер") is None
    assert classify("программа") is None


def test_classify_asr_error():
    tc = classify("двеси")
    assert tc and tc[0].value == 200


def test_classify_multiplier():
    tc = classify("тысячами")
    assert tc and tc[0].value == 1000 and tc[0].subtype == 'multiplier'


def test_classify_fused():
    tc = classify("дветысячи")
    assert tc and len(tc) == 2
    assert tc[0].value == 2 and tc[0].subtype == 'cardinal'
    assert tc[1].value == 1000 and tc[1].subtype == 'multiplier'


def test_classify_vague():
    tc = classify("тыщ", prev_tokens=["с", "чем", "то"])
    assert tc and tc[0].subtype == 'vague'


# ── Sequence parser tests ──


def test_sum_hundreds_tens():
    assert parse_sequence([TC(200, 3, 'cardinal'), TC(50, 2, 'cardinal')]) == ['250']


def test_enumeration_same_magnitude():
    assert parse_sequence([TC(200, 3, 'cardinal'), TC(300, 3, 'cardinal')]) == ['200', '300']


def test_sum_with_thousand_multiplier():
    assert parse_sequence([
        TC(2, 0, 'cardinal'), TC(1000, 4, 'multiplier'),
        TC(800, 3, 'cardinal'), TC(40, 2, 'cardinal'), TC(3, 0, 'cardinal')
    ]) == ['2843']


def test_two_multiplier_blocks():
    assert parse_sequence([
        TC(70, 2, 'cardinal'), TC(1000000, 5, 'multiplier'),
        TC(2, 0, 'cardinal'), TC(1000000, 5, 'multiplier')
    ]) == ['70000000', '2000000']


def test_standalone_thousand():
    assert parse_sequence([TC(1000, 4, 'multiplier')]) == ['1000']


def test_zero():
    assert parse_sequence([TC(0, 0, 'cardinal')]) == ['0']


def test_zero_and_seven():
    assert parse_sequence([TC(0, 0, 'cardinal'), TC(7, 0, 'cardinal')]) == ['0', '7']


def test_ordinal_in_compound():
    assert parse_sequence([
        TC(200, 3, 'cardinal'), TC(80, 1, 'cardinal'), TC(5, 0, 'ordinal')
    ]) == ['285']


def test_empty():
    assert parse_sequence([]) == []


def test_fused_compound():
    assert parse_sequence([
        TC(2, 0, 'cardinal'), TC(1000, 4, 'multiplier')
    ]) == ['2000']


def test_vague_skipped():
    assert parse_sequence([TC(1000, 4, 'vague')]) == []


def test_dva_dva_enum():
    assert parse_sequence([TC(2, 0, 'cardinal'), TC(2, 0, 'cardinal')]) == ['2', '2']


def test_compound_20_5():
    assert parse_sequence([TC(20, 2, 'cardinal'), TC(5, 0, 'cardinal')]) == ['25']


def test_large_multiplication():
    assert parse_sequence([
        TC(5, 0, 'cardinal'), TC(1000, 4, 'multiplier'),
        TC(200, 3, 'cardinal')
    ]) == ['5200']


def test_tens_enumeration():
    assert parse_sequence([TC(20, 2, 'cardinal'), TC(30, 2, 'cardinal')]) == ['20', '30']
