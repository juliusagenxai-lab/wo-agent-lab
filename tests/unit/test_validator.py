"""Unit tests for input validator and score clamper."""
import pytest

from src.scoring.validator import validate_inputs, clamp_level


class TestValidateInputs:
    def test_valid_inputs_return_no_errors(self):
        errors = validate_inputs(6, "argumentative", "Some essay text.")
        assert errors == []

    def test_invalid_grade_returns_error(self):
        errors = validate_inputs(5, "argumentative", "text")
        assert len(errors) == 1
        assert errors[0].field == "grade"

    def test_invalid_genre_returns_error(self):
        errors = validate_inputs(6, "persuasive", "text")
        assert len(errors) == 1
        assert errors[0].field == "genre"

    def test_empty_essay_returns_error(self):
        errors = validate_inputs(6, "argumentative", "")
        assert len(errors) == 1
        assert errors[0].field == "essay_text"

    def test_whitespace_only_essay_returns_error(self):
        errors = validate_inputs(6, "informative", "   \n  ")
        assert len(errors) == 1
        assert errors[0].field == "essay_text"

    def test_too_long_essay_returns_error(self):
        errors = validate_inputs(6, "narrative", "x" * 5001)
        assert len(errors) == 1
        assert errors[0].field == "essay_text"

    def test_multiple_errors_returned(self):
        errors = validate_inputs(5, "persuasive", "")
        assert len(errors) == 3

    def test_all_valid_genres(self):
        for genre in ("argumentative", "informative", "narrative"):
            assert validate_inputs(6, genre, "text") == []

    def test_all_valid_grades(self):
        for grade in (6, 7, 8):
            assert validate_inputs(grade, "argumentative", "text") == []


class TestClampLevel:
    def test_valid_levels_pass_through(self):
        for level in (1, 2, 3, 4):
            assert clamp_level(level) == level

    def test_above_max_clamped_to_4(self):
        assert clamp_level(5) == 4
        assert clamp_level(10) == 4

    def test_below_min_clamped_to_1(self):
        assert clamp_level(0) == 1
        assert clamp_level(-3) == 1

    def test_float_rounded(self):
        assert clamp_level(3.6) == 4
        assert clamp_level(2.4) == 2

    def test_invalid_input_defaults_to_1(self):
        assert clamp_level("bad") == 1
        assert clamp_level(None) == 1
