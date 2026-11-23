"""核验GPU末端迁移实验的性能与结果差异；保留所有轮次，不把波动当作加速。"""
import json
import math
import statistics
import sys
from pathlib import Path

folder = Path(sys.argv[1])
performance = {}
for mode in ['baseline', 'compact']:
    runs = [json.loads((folder/f'{mode}-r{i}.json').read_text(encoding='utf-8')) for i in [1, 2, 3]]
    for r in runs:
        assert r['seconds'] == 30 and r['degree'] == 4 and r['Mismatches'] == 0
        assert r['Completed'] == len(r['Rows'])
        assert math.isclose(r['ImagesPerSecond'], r['Completed']/30)
        assert all(x['Done'] <= 30 and x['Match'] and x['Ms'] > 0 for x in r['Rows'])
        assert not any('GpuOcrSession:alive=False' in s for s in r['states'])
    performance[mode] = dict(
        abba_mean_ips=statistics.mean(r['ImagesPerSecond'] for r in runs[:2]),
        abba_mean_cpu_cores=statistics.mean(r['CpuCores'] for r in runs[:2]),
        abba_mean_cpu_ms_per_image=statistics.mean(r['CpuMsPerImage'] for r in runs[:2]),
        rounds=[{k:r[k] for k in ['Completed','ImagesPerSecond','CpuCores','CpuMsPerImage','Mismatches']} for r in runs])
a, b = [json.loads((folder/f'corpus-{i}.json').read_text(encoding='utf-8')) for i in [0,1]]
a, b = [{r['File']:r for r in doc['Rows']} for doc in [a,b]]
assert a.keys() == b.keys() and len(a) == 90
text_changed=[]; boxes_changed=[]; alignment_changed=[]; score_changed=[]; max_score_difference=0
for name in a:
    assert a[name]['Success'] and b[name]['Success'] and a[name]['Stable'] and b[name]['Stable']
    assert a[name]['Sha256'] == b[name]['Sha256']
    x,y = a[name]['Result'],b[name]['Result']
    if x['Text'] != y['Text']: text_changed.append(name)
    if [l['Box'] for l in x['Lines']] != [l['Box'] for l in y['Lines']]: boxes_changed.append(name)
    def spans(result):
        return [[{k:v for k,v in s.items() if k != 'Score'} for s in l.get('CtcSpans') or []] for l in result['Lines']]
    if spans(x) != spans(y): alignment_changed.append(name)
    differences=[abs(l['RecognitionScore']-r['RecognitionScore']) for l,r in zip(x['Lines'],y['Lines'])]
    if any(differences): score_changed.append(name)
    max_score_difference=max([max_score_difference]+differences)
summary=dict(performance=performance,text_changed=text_changed,boxes_changed=boxes_changed,
             alignment_changed=alignment_changed,score_changed=score_changed,max_score_difference=max_score_difference)
(folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='performance'},ensure_ascii=False,indent=2))
for mode, r in performance.items():
    print(mode,round(r['abba_mean_ips'],2),'img/s',round(r['abba_mean_cpu_cores'],2),'CPU cores',round(r['abba_mean_cpu_ms_per_image'],2),'CPU ms/image')
