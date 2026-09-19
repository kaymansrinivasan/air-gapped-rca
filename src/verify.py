"""Conservative structured-evidence gate; not general causal entailment."""
import hashlib
from .ingest import safe_path


def verify(root, generated, bundle, retrieved):
    if not isinstance(generated, dict) or not isinstance(generated.get('candidates'), list):
        raise ValueError('Invalid generated schema')
    if len(generated['candidates']) > 3:
        raise ValueError('Too many candidates')
    accepted, rejected = [], []
    for candidate in generated['candidates']:
        try:
            if not isinstance(candidate, dict) or not isinstance(candidate.get('cause'), str) or not candidate['cause'].strip():
                raise ValueError('invalid_cause')
            basis = candidate.get('basis')
            if basis not in ('current_simulation', 'historical_hypothesis'):
                raise ValueError('invalid_basis')
            citations = candidate.get('citations')
            if not isinstance(citations, list) or not citations:
                raise ValueError('missing_citations')
            supports_cause = False
            for citation in citations:
                if not isinstance(citation, dict):
                    raise ValueError('invalid_citation')
                evidence_id = citation.get('evidence_id')
                if not isinstance(evidence_id, str) or evidence_id not in bundle['records']:
                    raise ValueError('unavailable_evidence')
                source = bundle['records'][evidence_id]
                if citation.get('path') != source['path']:
                    raise ValueError('path_mismatch')
                raw = safe_path(root, source['path']).read_bytes()
                if hashlib.sha256(raw).hexdigest() != source['sha256']:
                    raise ValueError('source_changed')
                start, end = citation.get('line_start'), citation.get('line_end')
                if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(source['lines']):
                    raise ValueError('invalid_line_range')
                if not any(c['evidence_id'] == evidence_id and c['line_start'] <= start <= end <= c['line_end'] for c in retrieved):
                    raise ValueError('citation_not_retrieved')
                quoted = '\n'.join(source['lines'][start-1:end])
                if citation.get('quote') != quoted:
                    raise ValueError('quote_mismatch')
                record = source['record']
                expected_role = 'current' if basis == 'current_simulation' else 'historical'
                if source['role'] != expected_role:
                    raise ValueError('basis_role_mismatch')
                if (record.get('cause_status') == 'CONFIRMED_WITHIN_SIMULATION_ONLY'
                    and record.get('interpretation_within_simulation') == candidate['cause']
                    and '"interpretation_within_simulation"' in quoted
                    and candidate['cause'] in quoted):
                    supports_cause = True
            if not supports_cause:
                raise ValueError('no_explicit_cause_support')
            # Reconstruct output: discard any extra, unverified model claims.
            accepted.append({'rank': len(accepted)+1, 'cause': candidate['cause'], 'basis': basis,
                             'real_world_cause_confirmed': False, 'citations': [
                                 {k: c[k] for k in ('evidence_id', 'path', 'line_start', 'line_end', 'quote')}
                                 for c in citations]})
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append({'reason': str(exc)})
    return {'status': 'answer' if accepted else 'refusal', 'candidates': accepted,
            'refusal_reason': None if accepted else 'No candidate passed the available-evidence policy.',
            'rejected_candidates': rejected,
            'scope': 'Unreviewed synthetic workflow demonstration; not a real-world diagnosis.'}
