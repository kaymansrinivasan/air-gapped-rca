import copy
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from shutil import copytree
from rca import run
from src.ingest import ingest, read_jsonl
from src.index import retrieve
from src.generate import fixture_generate, local_generate
from src.verify import verify

ROOT = Path(__file__).resolve().parents[1] / 'syn_data'
QUESTIONS = read_jsonl(ROOT / 'replay/questions.jsonl')


def question(suffix):
    return next(q for q in QUESTIONS if q['question_id'] == 'SYN-A595-' + suffix)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.q = question('001-Q-INVESTIGATION')
        self.bundle = ingest(ROOT, self.q)
        self.chunks = retrieve(self.bundle['chunks'], 'interpretation_within_simulation', 100)
        self.generated = fixture_generate(self.chunks)

    def test_valid_simulation_cause(self):
        result = verify(ROOT, self.generated, self.bundle, self.chunks)
        self.assertEqual(result['status'], 'answer')
        self.assertFalse(result['candidates'][0]['real_world_cause_confirmed'])

    def test_all_failure_stages_refuse_causes(self):
        for q in QUESTIONS:
            if q['stage'] == 'failure':
                with self.subTest(q=q['question_id']):
                    result = run(ROOT, q)
                    self.assertEqual(result['status'], 'refusal')
                    self.assertTrue(all(c['evidence_id'] in q['available_evidence_ids'] for c in result['retrieved']))

    def test_wrong_device_is_not_retrieved(self):
        result = run(ROOT, question('006-Q-RETEST'))
        self.assertEqual(result['rejected_evidence'][0]['reason'], 'device_mismatch')
        self.assertNotIn('SYN-A595-006-RETEST-1', [c['evidence_id'] for c in result['retrieved']])

    def test_future_citation_rejected(self):
        b = ingest(ROOT, question('001-Q-FAILURE'))
        result = verify(ROOT, self.generated, b, b['chunks'])
        self.assertEqual(result['status'], 'refusal')

    def test_mutated_citations_rejected(self):
        changes = [{'quote': 'fabricated quote'}, {'line_start': 0}, {'line_end': True},
                   {'path': '../reference/answer_key.jsonl'}, {'evidence_id': 'made-up'}]
        for change in changes:
            with self.subTest(change=change):
                g = copy.deepcopy(self.generated)
                g['candidates'][0]['citations'][0].update(change)
                self.assertEqual(verify(ROOT, g, self.bundle, self.chunks)['status'], 'refusal')

    def test_existing_quote_does_not_validate_wrong_cause(self):
        g = copy.deepcopy(self.generated)
        g['candidates'][0]['cause'] = 'An invented internal short'
        self.assertEqual(verify(ROOT, g, self.bundle, self.chunks)['status'], 'refusal')

    def test_unretrieved_citation_rejected(self):
        self.assertEqual(verify(ROOT, self.generated, self.bundle, [])['status'], 'refusal')

    def test_source_changed_after_ingestion(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Copy only indexed files; the pipeline does not read reference answers.
            copytree(ROOT / 'incidents', Path(tmp) / 'incidents')
            citation = self.generated['candidates'][0]['citations'][0]
            p = Path(tmp) / citation['path']
            p.write_text(p.read_text() + '\n', encoding='utf-8')
            self.assertEqual(verify(tmp, self.generated, self.bundle, self.chunks)['status'], 'refusal')

    def test_same_lineage_history_rejected(self):
        with self.assertRaises(ValueError):
            ingest(ROOT, self.q, ['SYN-A595-002'])

    def test_incompatible_history_is_excluded(self):
        b = ingest(ROOT, question('005-Q-FAILURE'), ['SYN-A595-001'])
        self.assertTrue(all(s['role'] == 'current' for s in b['records'].values()))
        self.assertEqual(b['rejected_evidence'][0]['reason'], 'historical_test_context_mismatch')

    def test_current_evidence_cannot_be_claimed_as_history(self):
        g = copy.deepcopy(self.generated)
        g['candidates'][0]['basis'] = 'historical_hypothesis'
        self.assertEqual(verify(ROOT, g, self.bundle, self.chunks)['status'], 'refusal')

    def test_answer_key_not_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            copytree(ROOT / 'incidents', Path(tmp) / 'incidents')
            for name in ('evidence_index.json', 'scenario_index.json'):
                (Path(tmp) / name).write_bytes((ROOT / name).read_bytes())
            self.assertEqual(run(Path(tmp), self.q)['status'], 'answer')

    def test_schema_error_is_not_refusal(self):
        with self.assertRaises(ValueError):
            verify(ROOT, {'candidates': 'bad'}, self.bundle, self.chunks)

    def test_external_endpoint_rejected_before_request(self):
        with self.assertRaises(ValueError):
            local_generate('https://example.com/v1', 'test', 'q', [])

    def test_local_adapter_contract(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                assert self.path == '/v1/chat/completions'
                assert payload['model'] == 'test-model'
                body = json.dumps({'choices': [{'message': {'content': '{"candidates": []}'}}]}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.assertEqual(local_generate(f'http://127.0.0.1:{server.server_port}/v1', 'test-model', 'q', []), {'candidates': []})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
