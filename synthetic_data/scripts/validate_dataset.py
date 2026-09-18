"""Validate source hashes, references, replay chronology, identity and test limits."""
from pathlib import Path
from collections import Counter
import json, hashlib, struct

ROOT=Path(__file__).resolve().parents[1]
def jl(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    for rel,digest in manifest['files'].items(): assert sha(ROOT/rel)==digest,('hash',rel)
    evidence=json.loads((ROOT/'evidence_index.json').read_text(encoding='utf-8'))
    scenarios=json.loads((ROOT/'scenario_index.json').read_text(encoding='utf-8'))
    index={s['incident_id']:s for s in scenarios}
    docs={eid:json.loads((ROOT/meta['path']).read_text(encoding='utf-8')) for eid,meta in evidence.items()}
    events=jl(ROOT/'replay/events.jsonl'); questions=jl(ROOT/'replay/questions.jsonl'); answers=jl(ROOT/'reference/answer_key.jsonl')
    text=(ROOT/'source/a595_tester_log.txt').read_text().splitlines()
    binary=(ROOT/'source/a595.stdf').read_bytes()
    real=0; full=0; partial=0; mismatch=[]; measurements=0
    for eid,doc in docs.items():
        assert doc['evidence_id']==eid
        scenario=index[doc['incident_id']]
        if doc['origin']=='REAL_OBSERVED':
            real+=1; t=doc['source_test_record']; c=doc['source_citation']
            assert doc['actual_physical_cause'] is None and t['root_cause'] is None
            assert sha(ROOT/c['file'])==c['sha256'] and sha(ROOT/c['binary_file'])==c['binary_sha256']
            assert text[c['line_start']-1]==t['raw_line']
            assert '\n'.join(text[c['line_start']-1:c['line_end']])==doc['excerpt']
            off=c['byte_offset']+4
            assert struct.unpack_from('>I',binary,off)[0]==t['test_number']
            assert binary[off+6]&128
        else:
            assert doc['origin']=='SIMULATED' and ('FICTIONAL' in doc['label'] or 'SIMULATED' in doc['label'])
            if 'tests' in doc:
                identity_matches=doc['synthetic_device_key']==scenario['synthetic_device_key']
                if not identity_matches: mismatch.append(eid)
                assert len(doc['tests'])==doc['test_count']
                assert len({t['test_number'] for t in doc['tests']})==doc['test_count']
                if doc['scope']=='FULL_CATALOGUE':
                    full+=1; assert doc['test_count']==123 and identity_matches
                    assert doc['device_outcome']=='PASS' and all(t['status']=='PASS' for t in doc['tests'])
                else:
                    partial+=1; assert doc['test_count']==1
                    assert doc['device_outcome']!='PASS'
                for t in doc['tests']:
                    measurements+=1; assert t['origin']=='SIMULATED'
                    src=t['template_provenance']; line=text[src['line_start']-1]
                    assert int(line.split()[0])==t['test_number']
                    if t['kind']=='parametric':
                        lower=t['value']>=t['lower_limit'] if t['lower_comparator']=='<=' else t['value']>t['lower_limit']
                        upper=t['value']<=t['upper_limit'] if t['upper_comparator']=='<=' else t['value']<t['upper_limit']
                        assert (lower and upper)==(t['status']=='PASS'),(eid,t['test_number'])
                    else: assert (t['failing_pin_count']==0)==(t['status']=='PASS')
        for cited in doc.get('cites',[]): assert cited in docs
    assert real==6 and full==3 and partial==4
    assert mismatch==['SYN-A595-006-RETEST-1']
    assert docs['SYN-A595-001-FAILURE']['source_test_record']==docs['SYN-A595-002-FAILURE']['source_test_record']
    assert docs['SYN-A595-001-INVESTIGATION']['interpretation_within_simulation']!=docs['SYN-A595-002-INVESTIGATION']['interpretation_within_simulation']
    bystage={}
    for incident in index:
        visible=[]; previous=-1
        for e in [x for x in events if x['incident_id']==incident]:
            assert e['replay_offset_seconds']>previous; previous=e['replay_offset_seconds']
            visible+=e['new_evidence_ids']
            assert visible==e['available_evidence_ids']
            for eid in e['new_evidence_ids']:
                assert docs[eid]['incident_id']==incident
                assert set(docs[eid].get('cites',[]))<=set(visible),('future citation',eid)
            bystage[incident,e['stage']]=set(visible)
        assert set(visible)==set(index[incident]['evidence_ids'])
    qmap={q['question_id']:q for q in questions}
    for q in questions: assert set(q['available_evidence_ids'])==bystage[q['incident_id'],q['stage']]
    for a in answers:
        assert set(a['required_citations'])<=set(qmap[a['question_id']]['available_evidence_ids'])
        assert a['actual_real_world_cause'] is None
    assert len(events)==24 and len(questions)==len(answers)==18
    result={'status':'PASS','incidents':6,'questions':18,'replay_events':24,'source_failure_copies':real,'full_synthetic_pass_retests':full,'targeted_synthetic_retests':partial,'synthetic_test_results':measurements,'intentional_identity_mismatches':mismatch,'future_evidence_citation_errors':0,'notice':'Structural consistency only; not physical-model validation or a model-accuracy score.'}
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
