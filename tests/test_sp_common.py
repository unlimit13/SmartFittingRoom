"""
R-11: 분산 가상 피팅(Mobile-VTON spatial-parallel) — row-band 분할 로직 단위 검증.

검증 대상은 src/ondevice_vton/parallel/sp_bands.py 의 band_bounds / band_size 다
(sp_common.py 가 그대로 re-export 하므로 `sp_common.band_bounds` 와 동일 로직).
각 rank가 소유하는 연속 row-band [h0:h1) 가
  (1) 결정적이고 (같은 입력 → 같은 출력),
  (2) 연속·무중첩으로 [0, H) 전체를 빠짐없이 덮으며,
  (3) 나누어떨어지지 않을 때 앞 rank가 +1을 가져가고,
  (4) world==2 에서 SP_BAND_FRAC0 로 비균등 분할이 가능
함을 확인한다.

분할 로직은 torch 비의존(sp_bands)으로 분리되어 있어, 메인 스위트
(requirements.txt, torch 미포함)에서도 실제로 실행·통과한다.
"""
import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src", "ondevice_vton", "parallel")
)
from sp_bands import band_bounds, band_size  # noqa: E402


def _bands(H, world):
    return [band_bounds(H, r, world) for r in range(world)]


@pytest.mark.parametrize("world", [2, 3, 4, 5])
@pytest.mark.parametrize("H", [16, 64, 96, 100, 128])
def test_R11_bands_cover_full_height_contiguously(H, world):
    """R-11: bands tile [0, H) with no gaps and no overlaps."""
    bands = _bands(H, world)
    assert bands[0][0] == 0
    assert bands[-1][1] == H
    for (h0, h1), (n0, _n1) in zip(bands, bands[1:]):
        assert h1 == n0          # contiguous, no gap / no overlap
    assert sum(h1 - h0 for h0, h1 in bands) == H


@pytest.mark.parametrize("world", [3, 4, 5])
@pytest.mark.parametrize("H", [100, 97, 128])
def test_R11_earlier_ranks_take_the_remainder(H, world):
    """R-11: when H % world != 0, the first `rem` ranks get one extra row."""
    base, rem = divmod(H, world)
    for rank in range(world):
        expected = base + (1 if rank < rem else 0)
        assert band_size(H, rank, world) == expected
    # sizes differ by at most 1 across ranks (balanced split)
    sizes = [band_size(H, r, world) for r in range(world)]
    assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize("world", [2, 4])
def test_R11_band_bounds_is_deterministic(world):
    """R-11: every rank computes the same bounds for any other rank, repeatably."""
    H = 120
    first = _bands(H, world)
    for _ in range(3):
        assert _bands(H, world) == first


def test_R11_world2_default_split_is_even():
    """R-11: 2-Pi default (SP_BAND_FRAC0 unset → 0.5) splits H in half."""
    os.environ.pop("SP_BAND_FRAC0", None)
    assert band_bounds(100, 0, 2) == (0, 50)
    assert band_bounds(100, 1, 2) == (50, 100)


def test_R11_world2_uneven_split_via_env():
    """R-11: SP_BAND_FRAC0 lets a Pi with less RAM own a smaller top band."""
    os.environ["SP_BAND_FRAC0"] = "0.3"
    try:
        h0_0, h1_0 = band_bounds(100, 0, 2)
        h0_1, h1_1 = band_bounds(100, 1, 2)
        assert (h0_0, h1_0) == (0, 30)
        assert (h0_1, h1_1) == (30, 100)
        # still contiguous and full-covering
        assert h1_0 == h0_1 and h1_1 == 100
    finally:
        os.environ.pop("SP_BAND_FRAC0", None)


def test_R11_band_size_matches_bounds():
    """R-11: band_size == h1 - h0 for every rank."""
    H, world = 113, 4
    for rank in range(world):
        h0, h1 = band_bounds(H, rank, world)
        assert band_size(H, rank, world) == h1 - h0
