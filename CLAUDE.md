# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 5. 평가 정합성 규율 (이 프로젝트 한정)

이 프로젝트는 평가 Agent(`project_guidelines/`)가 **요구사항 ↔ 코드 ↔ 테스트 ↔ 실행결과**의
추적 가능성으로 채점한다. 코드를 바꿀 때마다 이 사슬이 끊기지 않게 유지한다.
아래는 모든 코드 변경의 **완료 조건(Definition of Done)** 이다.

### 5.1 코드 변경에는 테스트가 동반된다
- `src/X.py`를 **추가하거나 동작을 바꾸면** 대응 테스트 `tests/test_X.py`를 같은 변경에서 추가·갱신한다.
  (배치는 `src/` 미러, 이름은 `test_*` 표준 패턴 — 가이드 §6의 자동 매칭 관례를 따른다.)
- **API·반환 계약이 바뀌면** 영향받는 테스트를 같은 변경에서 고친다.
  실패하거나 stale한 테스트를 **절대 남기지 않는다** (평가 Agent는 `pytest`를 실행해 통과/실패를 그대로 기록한다).
- 완료 전 전체 스위트가 green인지 확인한다: `pytest tests/`

### 5.2 테스트는 실제 코드 계약을 반영한다 (허구 mock 금지)
- mock 반환값은 실제 함수가 내놓는 구조와 **일치**해야 한다. mock이 옛 계약을 담은 채 통과하면
  테스트는 초록불이어도 실제 계약을 검증하지 못한다.
- 현행 핵심 계약 예시: `Recommender.recommend_outfit()` → **최대 3개(`NUM_CANDIDATES=3`) 코디 세트** 반환
  (`outfits: [{tops, bottoms, shoes}, ...]`, 각 슬롯은 `product_id/name/url/image_path/qr_b64`를 가진 상품 리스트).
  검색 후보가 부족하면 그보다 적은 수만 반환한다. `snap_id`·`anchor_score` 키는 없다.
  검출은 **MediaPipe Pose**(YOLO 아님). 라이브 피드는 `/detection_feed` (구 `/video_feed` 아님).

### 5.3 테스트 실행 결과 산출물(test-results)
- 제출·패키징 전 정본 결과를 재생성한다:
  `pytest tests/ --junitxml=test-results/junit.xml`
- `test-results/junit.xml`은 `결과파일.zip`에 포함되는 정본이다(가이드 §6 — JUnit XML은
  코드↔테스트↔결과 매칭의 최상위 신뢰 신호). `.gitignore`에서 이 파일만 예외로 추적한다.

### 5.4 실행 재현(RUN.md)·의존성 동기화
- 의존성을 바꾸면 `requirements.txt`의 **버전 고정**을 함께 갱신한다.
- 실행 방법·환경변수·외부 서비스(`FAL_KEY`, `VTON_BACKEND` 등)를 바꾸면 `RUN.md`의 DevOps 8개 기준을 갱신한다.

### 5.5 요구사항 명세서
- `deliverables/요구사항명세서.md`는 현재 **코드와 어긋난(outdated) 상태**다. 정답 기준으로 삼지 말 것.
- 별도 작업으로 **코드 기준으로 재작성**해야 한다. 재작성 시 각 요구사항에 중립 ID(`R-01`…)를 부여하고,
  해당 ID를 테스트 이름/도크스트링에 넣어(예: `test_R01_detection_feed_...`) 요구사항까지 한 줄로 추적되게 한다.
