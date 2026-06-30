"""Unit tests for the de-vig functions — the part of the CLV pipeline most worth pinning."""
import numpy as np
import pytest

from tennis_predictor.clv import key_tennisdata, keys_sackmann, proportional_devig, shin_devig


def test_shin_devig_is_complementary():
    """The two implied probabilities must sum to exactly 1 after de-vigging."""
    pw = shin_devig(1.5, 2.8)
    pl = shin_devig(2.8, 1.5)
    assert pw + pl == pytest.approx(1.0, abs=1e-6)


def test_shin_devig_removes_overround():
    """A de-vigged probability is below the raw 1/odds (the vig is stripped out)."""
    raw = 1 / 1.5
    assert shin_devig(1.5, 2.8) < raw


def test_shin_above_proportional_for_favourite():
    """Shin corrects favourite-longshot bias: it strips the underdog overbet, so the
    favourite's de-vigged probability is *higher* than proportional de-vig gives."""
    odds_fav, odds_dog = 1.3, 4.0
    assert shin_devig(odds_fav, odds_dog) > proportional_devig(odds_fav, odds_dog)


def test_proportional_devig_sums_to_one():
    pw = proportional_devig(1.5, 2.8)
    pl = proportional_devig(2.8, 1.5)
    assert pw + pl == pytest.approx(1.0)


def test_no_overround_falls_back_gracefully():
    """Fair odds (sum of inverses == 1) should return the proportional probability."""
    assert shin_devig(2.0, 2.0) == pytest.approx(0.5)


def test_name_keys_match_across_sources():
    """A Sackmann name and its tennis-data form should share a join key."""
    assert key_tennisdata("Alcaraz C.") in keys_sackmann("Carlos Alcaraz")
    assert key_tennisdata("Sabalenka A.") in keys_sackmann("Aryna Sabalenka")
