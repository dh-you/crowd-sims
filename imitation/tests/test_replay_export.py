import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_export import write_web_replay


def recording(layout="3x3", count=24, complete=True):
    return {
        "version": 2, "scene": "airplane", "layout": layout,
        "agent_count": count, "complete": complete,
        "width": 600, "height": 600, "fps": 12,
        "walls": [], "frames": [[[0, 100, 100, 3]], []] if complete
        else [[[0, 100, 100, 3]]],
    }


class ReplayExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.replays = self.root / "data" / "replays"

    def catalog(self):
        return json.loads((self.replays / "manifest.json").read_text())

    def test_equal_counts_in_different_layouts_get_distinct_files_and_entries(self):
        first = write_web_replay(recording("3x3"), self.root)
        second = write_web_replay(recording("2x2"), self.root)
        self.assertNotEqual(first, second)
        self.assertEqual(json.loads(first.read_text())["layout"], "3x3")
        self.assertEqual(json.loads(second.read_text())["layout"], "2x2")
        self.assertEqual(
            {(item["layout"], item["count"]) for item in self.catalog()["replays"]},
            {("3x3", 24), ("2x2", 24)},
        )

    def test_reexport_replaces_only_the_matching_layout_and_migrates_legacy_entry(self):
        self.replays.mkdir(parents=True)
        legacy = {"scene": "airplane", "count": 24, "file": "airplane-24.json", "frames": 1}
        queue = {"scene": "queue", "count": 3, "file": "queue-3.json", "frames": 1}
        (self.replays / "manifest.json").write_text(json.dumps({"version": 1, "replays": [legacy, queue]}))
        write_web_replay(recording("2x2"), self.root)
        write_web_replay(recording("3x3"), self.root)
        write_web_replay(recording("3x3"), self.root)
        entries = self.catalog()["replays"]
        self.assertEqual(len(entries), 3)
        self.assertIn(queue, entries)
        self.assertEqual(next(item for item in entries if item.get("layout") == "3x3")["file"],
                         "airplane-3x3-24.json")

    def test_explicit_destination_in_web_directory_is_registered(self):
        destination = self.replays / "custom-2x2.json"
        self.assertEqual(write_web_replay(recording("2x2"), self.root, destination), destination)
        self.assertEqual(self.catalog()["replays"][0]["file"], destination.name)

    def test_private_export_does_not_change_web_catalog(self):
        write_web_replay(recording(), self.root)
        before = (self.replays / "manifest.json").read_bytes()
        destination = self.root / "private" / "run.json"
        write_web_replay(recording("2x2"), self.root, destination)
        self.assertTrue(destination.exists())
        self.assertEqual((self.replays / "manifest.json").read_bytes(), before)

    def test_incomplete_export_preserves_previous_completed_recording_and_catalog(self):
        destination = write_web_replay(recording(), self.root)
        before_recording = destination.read_bytes()
        before_catalog = (self.replays / "manifest.json").read_bytes()
        diagnostic = write_web_replay(recording(complete=False), self.root)
        self.assertTrue(diagnostic.name.endswith(".incomplete.json"))
        self.assertFalse(json.loads(diagnostic.read_text())["complete"])
        self.assertEqual(destination.read_bytes(), before_recording)
        self.assertEqual((self.replays / "manifest.json").read_bytes(), before_catalog)

    def test_queue_exports_keep_existing_filename_and_catalog_shape(self):
        replay = recording()
        replay.update(scene="queue", agent_count=3)
        del replay["layout"]
        destination = write_web_replay(replay, self.root)
        self.assertEqual(destination.name, "queue-3.json")
        self.assertNotIn("layout", self.catalog()["replays"][0])


if __name__ == "__main__":
    unittest.main()
