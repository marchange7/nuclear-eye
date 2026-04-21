"""Unit tests for P4-7 Scout label → security intent mapping (any arch; runs in CI)."""

import unittest

from gesture_pose_mapping import (
    ALL_SCOUT_LABELS,
    APPLIANCE_ONLY_INTENTS,
    GESTURE_POSE_MAP,
    build_gesture_dict,
    map_gesture_pose,
)


class MapGesturePoseTests(unittest.TestCase):
    def test_hand_labels_match_scout_taxonomy(self) -> None:
        for label in ("thumbsUp", "openPalm", "pointUp", "peace", "fist", "unknown"):
            self.assertIn(label, GESTURE_POSE_MAP)

    def test_running_maps_to_fast_approach(self) -> None:
        intent, conf = map_gesture_pose("running", 1.0)
        self.assertEqual(intent, "fast_approach")
        self.assertGreater(conf, 0.99)

    def test_raised_hands_maps_to_hands_raised(self) -> None:
        intent, _ = map_gesture_pose("raised_hands", 0.8)
        self.assertEqual(intent, "hands_raised")

    def test_unknown_label_falls_back(self) -> None:
        intent, conf = map_gesture_pose("not_a_scout_label", 0.9)
        self.assertEqual(intent, "unknown")
        self.assertLessEqual(conf, 1.0)

    def test_build_gesture_dict_shape(self) -> None:
        d = build_gesture_dict("fist", 0.9, passthrough=False)
        self.assertEqual(d["source"], "scout_gesture_pose_mapping")
        self.assertEqual(d["intent"], "approaching")
        self.assertIn("confidence", d)
        self.assertNotIn("gesture_pose", d)

    def test_passthrough_includes_raw_label(self) -> None:
        d = build_gesture_dict("peace", 0.95, passthrough=True)
        self.assertEqual(d["gesture_pose"], "peace")

    def test_all_scout_labels_mapped(self) -> None:
        missing = ALL_SCOUT_LABELS - frozenset(GESTURE_POSE_MAP.keys())
        self.assertEqual(missing, frozenset(), f"GESTURE_POSE_MAP missing: {missing}")

    def test_appliance_only_intents_not_in_scout_map(self) -> None:
        mapped = {e.security_intent for e in GESTURE_POSE_MAP.values()}
        self.assertTrue(APPLIANCE_ONLY_INTENTS.isdisjoint(mapped))


if __name__ == "__main__":
    unittest.main()
