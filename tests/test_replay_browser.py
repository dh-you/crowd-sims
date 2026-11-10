"""Browser regressions for the replay selector; uses a local HTTP server."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import threading
import unittest
from urllib.parse import parse_qs, urlparse

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@unittest.skipIf(sync_playwright is None, "Install Playwright to run browser checks")
class ReplayBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]
        handler = partial(QuietHandler, directory=str(cls.repo))
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.runtime = sync_playwright().start()
        executable = shutil.which("chromium") or shutil.which("google-chrome")
        options = {"headless": True, "args": ["--no-sandbox"]}
        if executable:
            options["executable_path"] = executable
        try:
            cls.browser = cls.runtime.chromium.launch(**options)
        except Exception:
            cls.runtime.stop()
            cls.server.shutdown()
            cls.server.server_close()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.runtime.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1100, "height": 1000})
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.addCleanup(self.context.close)

    def open(self, query):
        self.page.goto(f"{self.base}/pages/replay.html?{query}")

    def wait_for_run(self, layout, count):
        self.page.wait_for_function(
            "([layout, count]) => !document.querySelector('#play').disabled && "
            "document.querySelector('#layout').value === layout && "
            "document.querySelector('#status').textContent.includes(count + ' agents')",
            arg=[layout, count],
        )

    def fixture(self, layouts=("3x3", "2x2"), legacy=False, incomplete=None):
        entries = [
            {"scene": "airplane", "count": 24, "file": f"fixture-{layout}-24.json",
             **({} if legacy else {"layout": layout})}
            for layout in layouts
        ]
        self.page.route(
            "**/data/replays/manifest.json",
            lambda route: route.fulfill(content_type="application/json", body=json.dumps(
                {"version": 1 if legacy else 2, "replays": entries})),
        )
        for layout in layouts:
            replay = {
                "scene": "airplane", "agent_count": 24, "width": 600, "height": 600,
                "complete": layout != incomplete, "fps": 0.01, "walls": [],
                "frames": [[[index, 100 + 10 * (index % 4), 100 + 10 * (index // 4), 3]
                            for index in range(24)], []],
                **({} if legacy else {"layout": layout}),
            }
            body = json.dumps(replay)
            def reply_with(payload):
                return lambda route: route.fulfill(content_type="application/json", body=payload)
            self.page.route(
                f"**/data/replays/fixture-{layout}-24.json",
                reply_with(body),
            )

    def test_all_checked_in_layouts_load_and_reach_an_empty_final_frame(self):
        self.open("scene=airplane&count=102")
        self.wait_for_run("3x3", 102)
        layouts = self.page.locator("#layout option").evaluate_all("options => options.map(o => o.value)")
        self.assertEqual(layouts, ["2x1", "2x2", "3x2", "3x3", "3x4", "3x4x3"])
        capacities = {"2x1": 51, "2x2": 68, "3x2": 85, "3x3": 102, "3x4": 119, "3x4x3": 170}
        for layout, count in capacities.items():
            with self.subTest(layout=layout):
                self.page.select_option("#layout", layout)
                self.wait_for_run(layout, count)
                self.assertIn("3–7", self.page.locator("#details").inner_text())
                self.page.locator("#timeline").evaluate(
                    "slider => { slider.value = slider.max; slider.dispatchEvent(new Event('input')); }"
                )
                self.assertIn("0 remaining", self.page.locator("#status").inner_text())
                self.page.click("#restart")
                self.assertIn(f"{count} remaining", self.page.locator("#status").inner_text())
        self.assertEqual(self.errors, [])

    def test_old_url_defaults_to_3x3_and_preserves_agent_count_selection(self):
        self.open("scene=airplane&count=102")
        self.wait_for_run("3x3", 102)
        self.assertEqual(self.page.locator("#count option").evaluate_all("options => options.map(o => o.value)"),
                         ["24", "60", "102"])
        self.page.select_option("#count", "24")
        self.wait_for_run("3x3", 24)
        self.assertEqual(parse_qs(urlparse(self.page.url).query)["layout"], ["3x3"])

    def test_widebody_deep_link_selects_two_aisles(self):
        self.open("scene=airplane&layout=3x4x3&count=170")
        self.wait_for_run("3x4x3", 170)
        self.assertIn("two aisles", self.page.locator("#layout option:checked").inner_text())
        self.assertIn("3–4–3 seating", self.page.locator("#status").inner_text())
        if os.environ.get("REPLAY_SCREENSHOT"):
            self.page.screenshot(path=os.environ["REPLAY_SCREENSHOT"], full_page=True)

    def test_queue_selection_hides_layout_and_removes_it_from_url(self):
        self.open("scene=airplane&layout=2x2&count=68")
        self.wait_for_run("2x2", 68)
        self.page.select_option("#scene", "queue")
        self.page.wait_for_function("() => !document.querySelector('#play').disabled && "
                                    "document.querySelector('#status').textContent.startsWith('queue')")
        self.assertTrue(self.page.locator("#layout-control").is_hidden())
        self.assertTrue(self.page.locator("#details").is_hidden())
        self.assertNotIn("layout", parse_qs(urlparse(self.page.url).query))
        self.assertEqual(self.errors, [])

    def test_equal_counts_select_the_correct_layout_file(self):
        self.fixture()
        self.open("scene=airplane&layout=2x2&count=24")
        self.wait_for_run("2x2", 24)
        self.assertIn("2–2 seating", self.page.locator("#status").inner_text())
        self.page.select_option("#layout", "3x3")
        self.wait_for_run("3x3", 24)
        self.assertIn("3–3 seating", self.page.locator("#status").inner_text())

    def test_legacy_catalog_and_recording_without_layout_still_load(self):
        self.fixture(layouts=("3x3",), legacy=True)
        self.open("scene=airplane&count=24")
        self.wait_for_run("3x3", 24)
        self.assertIn("3–3 seating", self.page.locator("#status").inner_text())
        self.assertEqual(self.errors, [])

    def test_older_slow_response_does_not_replace_latest_selection(self):
        self.fixture()
        self.page.add_init_script("""
            const originalFetch = window.fetch;
            window.fetch = async (...args) => {
                const response = await originalFetch(...args);
                if (String(args[0]).includes('fixture-2x2')) {
                    await new Promise(resolve => setTimeout(resolve, 700));
                }
                return response;
            };
        """)
        self.open("scene=airplane&layout=3x3&count=24")
        self.wait_for_run("3x3", 24)
        self.page.select_option("#layout", "2x2")
        self.page.select_option("#layout", "3x3")
        self.wait_for_run("3x3", 24)
        self.page.wait_for_timeout(1000)
        self.assertIn("3–3 seating", self.page.locator("#status").inner_text())
        self.assertEqual(parse_qs(urlparse(self.page.url).query)["layout"], ["3x3"])
        self.assertEqual(self.errors, [])

    def test_incomplete_recording_is_rejected_and_another_selection_recovers(self):
        self.fixture(incomplete="2x2")
        self.open("scene=airplane&layout=2x2&count=24")
        self.page.wait_for_function("() => document.querySelector('#status').textContent.includes('Could not load')")
        self.assertTrue(self.page.locator("#play").is_disabled())
        self.page.select_option("#layout", "3x3")
        self.wait_for_run("3x3", 24)


if __name__ == "__main__":
    unittest.main()
