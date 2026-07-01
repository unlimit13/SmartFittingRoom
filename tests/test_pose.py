"""
FR-03: 손 인식 커서 입력 — PoseTracker가 손 좌표와 zone 상태를 반환한다.
      기존 중앙 존 HOLD_SEC 트리거 로직은 현재 프론트엔드 자동 추천에는 쓰이지 않지만,
      PoseTracker 호환성 계약으로 유지 검증한다.

mediapipe 추론 결과(_pose.process)는 mock으로 대체하고, 유지 시간 경과는 진입 시각
(_in_zone_since)을 backdate 하여 결정적으로 재현한다. (실제 time.time()을 그대로 쓰므로
앱 테스트가 띄우는 백그라운드 pose 스레드와 간섭하지 않는다.)
"""
import os
import sys
import types
import unittest.mock as mock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("mediapipe")

from pose import PoseTracker, HOLD_SEC

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _fake_results(x=0.5, y=0.5, vis=1.0, n=33):
    """mediapipe 결과 흉내: n개 랜드마크 (x,y는 정규화 좌표, visibility)."""
    lm = [types.SimpleNamespace(x=x, y=y, visibility=vis) for _ in range(n)]
    return types.SimpleNamespace(pose_landmarks=types.SimpleNamespace(landmark=lm))


def _tracker(results):
    t = PoseTracker()
    t._pose = mock.MagicMock()
    t._pose.process.return_value = results
    return t


def test_no_landmarks_no_trigger():
    t = _tracker(types.SimpleNamespace(pose_landmarks=None))
    state = t.process(FRAME)
    assert state["joints"] is None
    assert state["boxes"] == {}
    assert state["in_zone"] is False
    assert state["triggered"] is False
    assert state["hold_pct"] == 0.0


def test_joints_outside_zone_not_in_zone():
    # 필수 관절이 화면 왼쪽 끝(존 밖)에 있음
    t = _tracker(_fake_results(x=0.01, y=0.5))
    state = t.process(FRAME)
    assert state["in_zone"] is False
    assert state["triggered"] is False


def test_trigger_after_hold():
    t = _tracker(_fake_results())  # 모든 관절이 중앙(존 안)

    s1 = t.process(FRAME)
    assert s1["in_zone"] is True
    assert s1["triggered"] is False

    # 진입 시각을 HOLD_SEC 이전으로 backdate → 다음 폴에서 유지 시간 충족
    t._in_zone_since -= HOLD_SEC + 1
    s2 = t.process(FRAME)
    assert s2["triggered"] is True
    assert s2["disabled"] is True


def test_no_retrigger_until_reset():
    t = _tracker(_fake_results())

    t.process(FRAME)                                  # 카운트 시작
    t._in_zone_since -= HOLD_SEC + 1
    assert t.process(FRAME)["triggered"] is True      # 트리거
    # disabled 상태 → 추가 폴은 재트리거하지 않음
    assert t.process(FRAME)["triggered"] is False
    t.reset()
    assert t.disabled is False


def _fake_body_results():
    """전신이 세로로 펼쳐진 랜드마크 흉내 (bbox가 0이 되지 않도록 x도 함께 분산)."""
    lm = [
        types.SimpleNamespace(x=0.45 + 0.1 * (i / 32), y=0.1 + 0.025 * i, visibility=1.0)
        for i in range(33)
    ]
    return types.SimpleNamespace(pose_landmarks=types.SimpleNamespace(landmark=lm))


def test_process_returns_normalized_hand_cursor_points():
    """FR-03 손 커서 입력: 왼손/오른손 좌표를 프레임 기준 [0,1]로 반환한다."""
    results = _fake_body_results()
    lm = results.pose_landmarks.landmark
    lm[19] = types.SimpleNamespace(x=0.25, y=0.30, visibility=1.0)
    lm[20] = types.SimpleNamespace(x=0.75, y=0.40, visibility=1.0)

    t = _tracker(results)
    state = t.process(FRAME)

    assert state["lw"] == pytest.approx([0.25, 0.30])
    assert state["rw"] == pytest.approx([0.75, 0.40])


def test_process_returns_region_boxes_for_full_body():
    """FR-02 라이브 시각화: 전신이 보이면 상의/하의/신발 박스 좌표가 함께 반환된다."""
    t = _tracker(_fake_body_results())
    state = t.process(FRAME)
    assert set(state["boxes"].keys()) <= {"tops", "bottoms", "shoes"}
    assert len(state["boxes"]) > 0
    for x1, y1, x2, y2 in state["boxes"].values():
        assert x2 > x1
        assert y2 > y1


def test_draw_overlay_skeleton_and_boxes_toggle():
    """FR-02 시각화 on/off: show_overlay=False면 스켈레톤·박스가 그려지지 않는다."""
    t = PoseTracker()
    state = {
        "joints": [(100, 100, 1.0)] * 33,
        "boxes": {"tops": (50, 50, 150, 150)},
        "rw": None,
        "lw": None,
    }

    frame_on = FRAME.copy()
    t.draw_overlay(frame_on, state, show_overlay=True)
    assert not np.array_equal(frame_on, FRAME)

    frame_off = FRAME.copy()
    t.draw_overlay(frame_off, state, show_overlay=False)
    assert np.array_equal(frame_off, FRAME)
