"""
Test Suite 1: Age Bracket Boundary Tests
Verifies that age categories are non-overlapping and test boundaries (17, 18, 35, 36, 50, 51, 65, 66)
as well as negative and invalid ages.
"""

import pytest
from app.analytics import get_age_bracket

def test_pediatric_boundary():
    assert get_age_bracket(0) == "Pediatric (<18)"
    assert get_age_bracket(17) == "Pediatric (<18)"

def test_young_adult_boundary():
    assert get_age_bracket(18) == "Young Adult (18-35)"
    assert get_age_bracket(35) == "Young Adult (18-35)"

def test_adult_boundary():
    assert get_age_bracket(36) == "Adult (36-50)"
    assert get_age_bracket(50) == "Adult (36-50)"

def test_mature_adult_boundary():
    assert get_age_bracket(51) == "Mature Adult (51-65)"
    assert get_age_bracket(65) == "Mature Adult (51-65)"

def test_senior_boundary():
    assert get_age_bracket(66) == "Senior (66+)"
    assert get_age_bracket(100) == "Senior (66+)"

def test_invalid_negative_age():
    with pytest.raises(ValueError):
        get_age_bracket(-1)

def test_invalid_none_age():
    with pytest.raises(ValueError):
        get_age_bracket(None)

