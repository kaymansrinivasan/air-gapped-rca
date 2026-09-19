"""Read indexed evidence only; retain immutable, UTF-8 source-line provenance."""
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Evidence path escapes dataset root')
    return path


def ingest(root, question, historical_incidents=(), chunk_lines=32):
    if chunk_lines < 1:
        raise ValueError('chunk_lines must be positive')
    root = Path(root)
    index = read_json(root / 'evidence_index.json')
    scenarios = {s['incident_id']: s for s in read_json(root / 'scenario_index.json')}
    current = scenarios[question['incident_id']]
    history = set(historical_incidents)
    for incident in history:
        if scenarios[incident]['source_lineage_group'] == current['source_lineage_group']:
            raise ValueError('Historical case shares current source lineage')
    allowed = set(question['available_evidence_ids'])
    if not allowed <= index.keys():
        raise ValueError('Unknown evidence ID in question')
    if any(index[e]['incident_id'] != current['incident_id'] for e in allowed):
        raise ValueError('Question includes evidence from another incident')
    # Conservative historical compatibility gate. Exact IDs alone are insufficient.
    def signature(incident):
        matches = [m for m in index.values() if m['incident_id'] == incident and m['stage'] == 'failure']
        if len(matches) != 1:
            return None
        record = read_json(safe_path(root, matches[0]['path'])).get('source_test_record', {})
        values = tuple(record.get(k) for k in ('program', 'test_name', 'kind'))
        return values if all(values) else None
    current_signature = signature(current['incident_id'])
    incompatible = {i for i in history if current_signature is None or signature(i) != current_signature}
    history -= incompatible
    selected = allowed | {e for e, meta in index.items() if meta['incident_id'] in history}
    records, chunks = {}, []
    rejected = [{'incident_id': i, 'reason': 'historical_test_context_mismatch'} for i in sorted(incompatible)]
    for evidence_id in sorted(selected):
        meta = index[evidence_id]
        # The allowlist cannot be redirected to answer keys or unrelated files.
        if not meta['path'].startswith('incidents/' + meta['incident_id'] + '/'):
            raise ValueError('Evidence must reside in its incident directory')
        path = safe_path(root, meta['path'])
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        record = json.loads(text)
        scenario = scenarios[meta['incident_id']]
        if record['evidence_id'] != evidence_id or record['incident_id'] != meta['incident_id']:
            raise ValueError('Evidence identity mismatch')
        if record['origin'] != meta['origin']:
            raise ValueError('Evidence origin mismatch')
        if record['synthetic_device_key'] != scenario['synthetic_device_key']:
            rejected.append({'evidence_id': evidence_id, 'reason': 'device_mismatch'})
            continue
        role = 'current' if meta['incident_id'] == current['incident_id'] else 'historical'
        records[evidence_id] = {'record': record, 'path': meta['path'], 'role': role,
                                'sha256': hashlib.sha256(raw).hexdigest(), 'lines': text.splitlines()}
        lines = text.splitlines()
        for start in range(0, len(lines), chunk_lines):
            chunks.append({'evidence_id': evidence_id, 'path': meta['path'], 'role': role,
                           'origin': record['origin'], 'sha256': records[evidence_id]['sha256'],
                           'line_start': start + 1, 'line_end': min(start + chunk_lines, len(lines)),
                           'text': '\n'.join(lines[start:start + chunk_lines])})
    return {'records': records, 'chunks': chunks, 'rejected_evidence': rejected}
