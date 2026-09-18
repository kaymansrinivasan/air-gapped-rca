"""Emit stage-limited synthetic RCA evidence. No model inference is performed."""
from pathlib import Path
import argparse, json, time, sys

ROOT=Path(__file__).resolve().parents[1]
def jl(path): return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--incident',default='SYN-A595-001')
    p.add_argument('--speed',type=float,default=0,help='0: instant; positive: simulated seconds per wall-clock second')
    p.add_argument('--show-reference',action='store_true',help='Display designer reference answers; never send them to the model under test')
    args=p.parse_args()
    if args.speed<0: p.error('--speed must be zero or positive')
    events=[e for e in jl(ROOT/'replay/events.jsonl') if e['incident_id']==args.incident]
    if not events: p.error('Unknown incident ID')
    evidence=json.loads((ROOT/'evidence_index.json').read_text(encoding='utf-8'))
    questions={q['stage']:q for q in jl(ROOT/'replay/questions.jsonl') if q['incident_id']==args.incident}
    references={a['question_id']:a for a in jl(ROOT/'reference/answer_key.jsonl')} if args.show_reference else {}
    last=0
    for e in events:
        if args.speed: time.sleep((e['replay_offset_seconds']-last)/args.speed)
        last=e['replay_offset_seconds']
        docs=[json.loads((ROOT/evidence[i]['path']).read_text(encoding='utf-8')) for i in e['new_evidence_ids']]
        print(json.dumps({'record_type':'EVIDENCE_EVENT','notice':'SYNTHETIC WORKFLOW DEMO; source failure records retain REAL_OBSERVED provenance.','event':e,'new_documents':docs}),flush=True)
        q=questions.get(e['stage'])
        if q:
            print(json.dumps({'record_type':'QUESTION','question':q}),flush=True)
            if args.show_reference:
                print(json.dumps({'record_type':'DESIGNER_REFERENCE_NOT_MODEL_OUTPUT','reference':references[q['question_id']]}),flush=True)
if __name__=='__main__': main()
