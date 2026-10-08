import json
from pathlib import Path
import unittest

from src.rca_local.core import EvidenceError, observation_from_form, adapt_procedure
from src.rca_local.inputs import TESTS, inspect_log, form_options

ROOT = Path(__file__).resolve().parents[1]


class InputTests(unittest.TestCase):
    def form(self, **changes):
        return dict(product='Product_A', failed_test=210,
                    question='What checks would distinguish possible causes?', **changes)

    def test_log_and_pins_optional_without_fabricated_measurements(self):
        current = observation_from_form(self.form())
        self.assertFalse(current['dut_id_provided'])
        self.assertFalse(current['index_as_history'])
        self.assertEqual(current['observation_text'], '')
        self.assertEqual(current['test_result']['failing_pins'], [])
        self.assertIn('measurements and investigation results are unknown', current['retrieval_text'])
        self.assertNotIn(current['dut_id'], current['retrieval_text'])
        self.assertEqual(adapt_procedure('Measure D100.', 'Product_A-L01-W01-D100', ''), 'Measure the current DUT.')

    def test_all_historical_test_ids_can_be_selected_and_extracted(self):
        found = set()
        for path in sorted((ROOT / 'syn_data/product_a_scenarios_v1').glob('case_*/observation.json')):
            record = json.loads(path.read_text())
            if record.get('role') != 'historical':
                continue
            expected = record['original_chunk']['failed_test']
            with self.subTest(path=path):
                hints = inspect_log(record['retrieval_text'])
                self.assertEqual(hints['failed_test'], expected)
                current = observation_from_form(dict(product='Product_A', observation=record['retrieval_text'], question='What should I check?'))
                self.assertEqual(current['test_result']['test_number'], expected)
                self.assertEqual(current['dut_id'], record['dut_id'])
                found.add(expected)
        self.assertEqual(found, set(TESTS))

    def test_observed_continuity_vs_idd_conflict_is_blocked(self):
        form = self.form(observation='Continuity (100): PASS.\nIDD_Static (210): FAIL.\nScan (606): NOT RUN.')
        form['failed_test'] = 100
        with self.assertRaisesRegex(EvidenceError, 'conflicts with the log'):
            observation_from_form(form)

    def test_unsupported_id_is_not_silently_translated(self):
        with self.assertRaisesRegex(EvidenceError, 'unsupported failed-test number 200'):
            inspect_log('IDD_Static (200): FAIL — above limit.')

    def test_dut_and_tester_conflicts_and_multiple_duts(self):
        text = 'DUT: Product_A-L06-W02-D008\nTester: ate-01 Program: product_a_ws\nIDD_Static (210): FAIL'
        for key, value in [('dut_id', 'OTHER-DUT'), ('tester', 'ate-02'), ('program', 'different')]:
            with self.subTest(key=key), self.assertRaises(EvidenceError):
                observation_from_form(self.form(observation=text, **{key:value}))
        with self.assertRaisesRegex(EvidenceError, 'multiple DUT'):
            inspect_log(text+'\nDUT: Product_A-L06-W02-D009')

    def test_not_run_and_question_are_not_failure_evidence(self):
        self.assertNotIn('failed_test', inspect_log('Scan (606): NOT RUN.'))
        with self.assertRaises(EvidenceError):
            observation_from_form(dict(product='Product_A', question='Did Scan fail?'))

    def test_choices_are_derived_from_local_history(self):
        options = form_options(ROOT)
        self.assertEqual([t['number'] for t in options['tests']], [100,210,606])
        self.assertIn('ate-01', options['testers'])
        self.assertIn('product_a_ws', options['programs'])


if __name__ == '__main__':
    unittest.main()
