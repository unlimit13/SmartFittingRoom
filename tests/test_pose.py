"""
R-12: 포즈 기반 자동 추천 트리거 — 필수 관절이 화면 중앙 존 안에 HOLD_SEC 유지되면 trigger.
      트리거 후 zone은 disabled 되고, reset()으로만 재활성화된다.

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
