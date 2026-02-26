"""
Simple real-time fitness coach for the Live Stream tab.

Goal:
- Count reps (squats, push-ups) from pose angles.
- Emit short corrective cues that can be shown on-screen and spoken via TTS in the browser.

This is intentionally heuristic-based (fast + robust) rather than a heavy action model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


SUPPORTED_EXERCISES = ("auto", "squat", "pushup", "lunge", "plank", "jumpingjack", "shoulderpress")


@dataclass
class CoachState:
    exercise: str = "auto"
    reps: int = 0
    stage: str = "unknown"  # up | down | hold | unknown
    last_cue: str = ""
    last_angles: Dict[str, float] = field(default_factory=dict)
    hold_start: float = 0.0
    streak: int = 0  # consecutive good-form reps
    best_streak: int = 0


def _best_side(angles: Dict[str, float], left_key: str, right_key: str) -> float:
    lv = angles.get(left_key, 0.0) or 0.0
    rv = angles.get(right_key, 0.0) or 0.0
    if lv <= 0 and rv > 0:
        return rv
    if rv <= 0 and lv > 0:
        return lv
    return max(lv, rv)


def _time_now() -> float:
    import time
    return time.monotonic()


class FitnessCoach:
    def __init__(self, exercise: str = "auto"):
        self.state = CoachState(exercise=exercise)

    def set_exercise(self, exercise: str):
        if exercise in SUPPORTED_EXERCISES:
            if exercise != self.state.exercise:
                self.state.exercise = exercise
                self.state.reps = 0
                self.state.stage = "unknown"
                self.state.streak = 0

    def update(self, angles: Dict[str, float]) -> Dict:
        """
        Update coach state from a pose angle snapshot.

        Returns a JSON-serializable dict with exercise data, motivational cues,
        streak tracking, and form quality scoring.
        """
        st = self.state
        st.last_angles = angles or {}

        knee = _best_side(angles, "left_knee", "right_knee")
        elbow = _best_side(angles, "left_elbow", "right_elbow")

        exercise = st.exercise
        if exercise == "auto":
            if elbow and elbow < 120:
                exercise = "pushup"
            elif knee and knee < 140:
                exercise = "squat"
            else:
                exercise = "squat"

        cue = ""
        score = 0.75
        motivational = ""

        if exercise == "squat":
            down_th, up_th = 105, 165
            if knee <= 0:
                cue = "I can't see your legs clearly. Step back and ensure full body is in frame."
                score = 0.4
            else:
                if knee < 90:
                    cue = "Very deep squat! Keep your chest up and knees tracking over toes."
                    score = 0.85
                elif knee < 115:
                    cue = "Good depth. Drive through your heels to stand up."
                    score = 0.9
                elif knee < 140:
                    cue = "Go a bit deeper — aim to get hips below knee level."
                    score = 0.65
                else:
                    cue = "Stand tall at the top. Control your descent on the way down."
                    score = 0.75

                if st.stage in ("unknown", "up") and knee < down_th:
                    st.stage = "down"
                elif st.stage == "down" and knee > up_th:
                    st.stage = "up"
                    st.reps += 1
                    if score >= 0.8:
                        st.streak += 1
                        st.best_streak = max(st.best_streak, st.streak)
                    else:
                        st.streak = 0

        elif exercise == "pushup":
            down_th, up_th = 100, 160
            if elbow <= 0:
                cue = "I can't see your arms clearly. Move sideways and keep shoulders/elbows visible."
                score = 0.4
            else:
                if elbow < 85:
                    cue = "Great depth! Keep your core tight — no sagging hips."
                    score = 0.9
                elif elbow < 110:
                    cue = "Go a little lower. Keep elbows about 45° from your torso."
                    score = 0.75
                elif elbow < 145:
                    cue = "Press up fully at the top. Maintain a straight line head-to-heels."
                    score = 0.7
                else:
                    cue = "Lock out gently at the top. Maintain a solid plank position."
                    score = 0.78

                if st.stage in ("unknown", "up") and elbow < down_th:
                    st.stage = "down"
                elif st.stage == "down" and elbow > up_th:
                    st.stage = "up"
                    st.reps += 1
                    if score >= 0.8:
                        st.streak += 1
                        st.best_streak = max(st.best_streak, st.streak)
                    else:
                        st.streak = 0

        elif exercise == "lunge":
            down_th, up_th = 100, 155
            if knee <= 0:
                cue = "Step into the lunge. I need to see your legs clearly."
                score = 0.4
            else:
                if knee < 90:
                    cue = "Deep lunge! Keep your front knee over your ankle, not past your toes."
                    score = 0.85
                elif knee < 110:
                    cue = "Good lunge depth. Keep your torso upright and core engaged."
                    score = 0.9
                elif knee < 140:
                    cue = "Lower your back knee closer to the ground for a deeper lunge."
                    score = 0.6
                else:
                    cue = "Step forward into position. Feet should be hip-width apart."
                    score = 0.7

                if st.stage in ("unknown", "up") and knee < down_th:
                    st.stage = "down"
                elif st.stage == "down" and knee > up_th:
                    st.stage = "up"
                    st.reps += 1
                    if score >= 0.8:
                        st.streak += 1
                        st.best_streak = max(st.best_streak, st.streak)
                    else:
                        st.streak = 0

        elif exercise == "plank":
            if elbow <= 0 and knee <= 0:
                cue = "Get into plank position. I need to see your full body from the side."
                score = 0.4
            else:
                now = _time_now()
                body_straight = (knee > 155 if knee else True) and (elbow > 155 if elbow else True)
                if body_straight:
                    if st.stage != "hold":
                        st.stage = "hold"
                        st.hold_start = now
                    hold_secs = now - st.hold_start
                    if hold_secs < 10:
                        cue = f"Good form! Hold steady. {int(hold_secs)}s — keep going!"
                        score = 0.85
                    elif hold_secs < 30:
                        cue = f"Strong hold at {int(hold_secs)}s! Squeeze your glutes and breathe."
                        score = 0.9
                    else:
                        cue = f"Incredible! {int(hold_secs)}s plank! You're a machine!"
                        score = 0.95
                    st.reps = max(st.reps, int(hold_secs))
                else:
                    cue = "Your body is sagging. Tighten your core and keep a straight line."
                    score = 0.55
                    st.stage = "unknown"

        elif exercise == "jumpingjack":
            if knee <= 0 and elbow <= 0:
                cue = "Start jumping jacks! Make sure your full body is visible."
                score = 0.4
            else:
                arms_up = (elbow > 150) if elbow else False
                legs_apart = (knee > 160) if knee else False

                if arms_up and legs_apart:
                    if st.stage != "up":
                        st.stage = "up"
                    cue = "Arms up, legs wide — great form! Now bring them back in."
                    score = 0.85
                elif not arms_up and not legs_apart:
                    if st.stage == "up":
                        st.stage = "down"
                        st.reps += 1
                        st.streak += 1
                        st.best_streak = max(st.best_streak, st.streak)
                    cue = "Jump out again! Keep the rhythm going."
                    score = 0.8
                else:
                    cue = "Coordinate arms and legs together. Jump out fully!"
                    score = 0.6

        elif exercise == "shoulderpress":
            up_th, down_th = 155, 90
            if elbow <= 0:
                cue = "Hold weights at shoulder height. I need to see your arms."
                score = 0.4
            else:
                if elbow > up_th:
                    cue = "Full extension! Don't lock out completely — maintain tension."
                    score = 0.85
                    if st.stage == "down":
                        st.stage = "up"
                        st.reps += 1
                        if score >= 0.8:
                            st.streak += 1
                            st.best_streak = max(st.best_streak, st.streak)
                        else:
                            st.streak = 0
                elif elbow < down_th:
                    cue = "Lower the weights to shoulder level. Control the descent."
                    score = 0.75
                    st.stage = "down"
                else:
                    cue = "Press upward! Drive through your shoulders."
                    score = 0.7
                    if st.stage == "unknown":
                        st.stage = "down"

        else:
            cue = "Select an exercise to begin coaching."
            score = 0.5

        # Motivational messages based on streaks
        if st.streak >= 10:
            motivational = "UNSTOPPABLE! 10+ perfect reps in a row!"
        elif st.streak >= 5:
            motivational = "On fire! 5+ perfect form streak!"
        elif st.streak >= 3:
            motivational = "Great consistency! Keep it up!"
        elif st.reps > 0 and st.reps % 10 == 0:
            motivational = f"Milestone! {st.reps} reps completed!"

        if motivational:
            cue = f"{cue} {motivational}"

        st.last_cue = cue

        return {
            "exercise": exercise,
            "reps": st.reps,
            "stage": st.stage,
            "cue": cue,
            "angles_used": {"knee": round(knee, 1), "elbow": round(elbow, 1)},
            "score": round(score, 2),
            "streak": st.streak,
            "best_streak": st.best_streak,
        }

