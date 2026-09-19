"""Replay plumbing checks. Fixture outcomes are never reported as RCA accuracy."""
import argparse
import json
from pathlib import Path
from rca import run
from src.ingest import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('syn_data'))
    parser.add_argument('--mode', choices=['fixture', 'local'], default='fixture')
    parser.add_argument('--endpoint', default='http://127.0.0.1:8100/v1')
    parser.add_argument('--model')
    args = parser.parse_args()
    counts = {'answer': 0, 'refusal': 0, 'error': 0}
    for q in read_jsonl(args.data / 'replay/questions.jsonl'):
        try:
            result = run(args.data, q, args.mode, args.endpoint, args.model)
        except Exception as exc:
            result = {'question_id': q['question_id'], 'status': 'error', 'error': str(exc)}
        counts[result['status']] += 1
        print(json.dumps(result))
    print(json.dumps({'summary': counts, 'mode': args.mode,
                      'accuracy': None, 'false_answer_rate': None,
                      'notice': 'Unreviewed demonstration replay; no ground-truth scoring or board qualification.'}))
    if counts['error']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
