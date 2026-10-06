import json
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

from src.rca_local import web


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def test_page_example_and_host_guard(self):
        with urlopen(self.url) as response:
            page = response.read().decode()
            self.assertIn("Investigation desk", page)
            self.assertIn(web.TOKEN, page)
            self.assertNotIn("__RCA_TOKEN__", page)
        with urlopen(self.url + "/api/example") as response:
            self.assertEqual(json.load(response)["dut_id"], "Product_A-L06-W01-D001")
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.url, headers={"Host": "attacker.example"}))
        self.assertEqual(caught.exception.code, 403)

    def test_csrf_and_input_validation(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.url + "/api/analyze", data=b"{}", method="POST"))
        self.assertEqual(caught.exception.code, 403)
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.url + "/api/analyze", data=b"{}", headers={"X-RCA-Token": web.TOKEN}, method="POST"))
        self.assertEqual(caught.exception.code, 400)

    def test_form_to_background_result(self):
        data = {"dut_id": "NEW-DUT", "product": "Product_A", "failed_test": 100,
                "observation": "Continuity failed", "question": "What should I check?"}
        with patch.object(web, "run_pipeline", return_value={"status": "refuse", "summary": "Test response"}) as pipeline:
            with urlopen(Request(self.url + "/api/analyze", data=json.dumps(data).encode(), headers={"X-RCA-Token": web.TOKEN}, method="POST")) as response:
                job = json.load(response)["job_id"]
            for _ in range(30):
                with urlopen(self.url + "/api/jobs/" + job) as response:
                    state = json.load(response)
                if state["state"] == "done":
                    break
                time.sleep(.01)
            self.assertEqual(state["answer"]["status"], "refuse")
            self.assertEqual(pipeline.call_args.args[0]["dut_id"], "NEW-DUT")


if __name__ == "__main__":
    unittest.main()
