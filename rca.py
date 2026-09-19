"""Run one staged question: python rca.py --help."""
import argparse
import json
from pathlib import Path
from src.ingest import ingest, read_jsonl
from src.index import retrieve
from src.generate import fixture_generate, local_generate
from src.verify import verify
from src.measure import timed


def run(root, question, mode='fixture', endpoint='http://127.0.0.1:8100/v1', model=None, history=(), top_k=12):
    timings = {}
    bundle, timings['ingest_seconds'] = timed(ingest, root, question, history)
    # Enrich generic questions with current observed test identity, never later events.
    failures = [s['record'].get('source_test_record', {}) for s in bundle['records'].values()
                if s['role'] == 'current' and s['record'].get('origin') == 'REAL_OBSERVED']
    query = question['question'] + ' interpretation_within_simulation cause_status ' + json.dumps(failures)
    retrieved, timings['retrieval_seconds'] = timed(retrieve, bundle['chunks'], query, top_k)
    if mode == 'fixture':
        generated, timings['generation_seconds'] = timed(fixture_generate, retrieved)
    elif mode == 'local':
        if not model:
            raise ValueError('--model is required for local mode')
        generated, timings['generation_seconds'] = timed(local_generate, endpoint, model, question['question'], retrieved)
    else:
        raise ValueError('Unknown mode')
    result, timings['verification_seconds'] = timed(verify, root, generated, bundle, retrieved)
    result.update(question_id=question['question_id'], incident_id=question['incident_id'], mode=mode,
                  timings=timings, retrieved=retrieved, generated=generated,
                  rejected_evidence=bundle['rejected_evidence'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('syn_data'))
    parser.add_argument('--question-id', required=True)
    parser.add_argument('--mode', choices=['fixture', 'local'], default='fixture')
    parser.add_argument('--endpoint', default='http://127.0.0.1:8100/v1')
    parser.add_argument('--model')
    parser.add_argument('--history', nargs='*', default=[])
    parser.add_argument('--top-k', type=int, default=12)
    args = parser.parse_args()
    try:
        questions = read_jsonl(args.data / 'replay/questions.jsonl')
        question = next(q for q in questions if q['question_id'] == args.question_id)
        result = run(args.data, question, args.mode, args.endpoint, args.model, args.history, args.top_k)
    except Exception as exc:
        print(json.dumps({'status': 'error', 'error': str(exc) or type(exc).__name__}))
        raise SystemExit(1)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
