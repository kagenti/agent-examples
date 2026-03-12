"""Tests for flight_tool MCP server — pure utility functions (isolated from heavy deps)."""

import sys
from datetime import date, timedelta
from unittest.mock import MagicMock

# Mock the fastmcp and fast_flights dependencies before importing
sys.modules.setdefault("fastmcp", MagicMock())
sys.modules.setdefault("fast_flights", MagicMock())

from flight_tool import _coerce_int, _date_in_past, _parse_iso_date, _result_to_dict


class TestParseIsoDate:
    """Test _parse_iso_date helper."""

    def test_valid_date(self):
        result = _parse_iso_date("2025-06-15")
        assert result == date(2025, 6, 15)

    def test_invalid_format(self):
        assert _parse_iso_date("15/06/2025") is None

    def test_empty_string(self):
        assert _parse_iso_date("") is None


class TestDateInPast:
    """Test _date_in_past helper."""

    def test_past_date(self):
        past = date.today() - timedelta(days=1)
        assert _date_in_past(past) is True

    def test_future_date(self):
        future = date.today() + timedelta(days=1)
        assert _date_in_past(future) is False

    def test_today(self):
        assert _date_in_past(date.today()) is False


class TestCoerceInt:
    """Test _coerce_int helper."""

    def test_int_value(self):
        val, err = _coerce_int(5, "adults", 1)
        assert val == 5
        assert err is None

    def test_string_value(self):
        val, err = _coerce_int("3", "adults", 1)
        assert val == 3
        assert err is None

    def test_string_with_spaces(self):
        val, err = _coerce_int(" 2 ", "adults", 1)
        assert val == 2
        assert err is None

    def test_invalid_string(self):
        val, err = _coerce_int("abc", "adults", 1)
        assert val == 1
        assert err is not None
        assert "Invalid integer" in err

    def test_negative_value(self):
        val, err = _coerce_int(-1, "adults", 1)
        assert val == 1
        assert err is not None
        assert "must be >= 0" in err

    def test_zero_value(self):
        val, err = _coerce_int(0, "children", 0)
        assert val == 0
        assert err is None

    def test_invalid_type(self):
        val, err = _coerce_int(3.5, "adults", 1)
        assert val == 1
        assert err is not None
        assert "Invalid type" in err


class TestResultToDict:
    """Test _result_to_dict helper."""

    def test_empty_result(self):
        class MockResult:
            flights = []

        result = _result_to_dict(MockResult())
        assert len(result) == 1
        assert result[0]["id"] is None
        assert result[0]["price"] == "N/A"

    def test_result_with_flights(self):
        class MockFlight:
            name = "UA123"
            duration = 180
            stops = 0
            departure = "2025-06-15T08:00"
            arrival = "2025-06-15T11:00"
            is_best = True
            delay = None

        class MockResult:
            flights = [MockFlight()]
            current_price = "$250"

        result = _result_to_dict(MockResult())
        assert len(result) == 1
        assert result[0]["id"] == "UA123"
        assert result[0]["airline"] == "UA123"
        assert result[0]["is_best"] is True
        assert result[0]["stops"] == 0

    def test_no_flights_attribute(self):
        class MockResult:
            pass

        result = _result_to_dict(MockResult())
        assert len(result) == 1
        assert result[0]["id"] is None
