"""核验并发扫描，保留每轮数据并区分延迟、吞吐及GPU时间戳口径。"""
import json
import math
import re
import statistics
import sys
from pathlib import Path

root=Path(sys.argv[1])
rows=[]
for mode in ['A','B','C']:
    for degree in [1,2,4,6,8]:
        for repeat in [1,2]:
            name=f'{mode}-c{degree}-r{repeat}'
            r=json.loads((root/f'{name}.json').read_text(encoding='utf-8'))
            assert r['degree']==degree and r['seconds']==12 and not r['profile'] and r['decode']
            assert r['WarmupSeconds']==8 and not r['GpuCls'] and not r['cpu'] and r['ctc']==0 and r['prep']==0
            assert r['Mismatches']==0 and all(x['Match'] and x['Done']<=12 for x in r['Rows'])
            assert r['Completed']==len(r['Rows']) and math.isclose(r['ImagesPerSecond'],len(r['Rows'])/12)
            assert r['CompactHead']==('0' if mode=='A' else '1')
            assert r['SingleWave']==('1' if mode=='C' else '0')
            assert not any('alive=False' in s for s in r['states'])
            assert not re.search(r'falling back|ErrorDeviceLost', (root/f'{name}.log').read_text(encoding='utf-8'),re.I)
            rows.append(dict(name=name,mode=mode,degree=degree,repeat=repeat,raw=r))
assert len({x['raw']['BinarySha256'] for x in rows})==1
assert all(x['raw']['Images']==rows[0]['raw']['Images'] for x in rows)
reference={}
for x in rows:
    for image,result in zip(x['raw']['Images'],x['raw']['Baselines']):
        # All warmed requests within a mode must retain the same complete DTO.
        key=x['mode'],image['File']
        assert reference.setdefault(key,result)==result,(x['name'],image['File'])
for image in rows[0]['raw']['Images']:
    assert reference['B',image['File']]==reference['C',image['File']],image['File']
groups={}
for mode in ['A','B','C']:
    groups[mode]={}
    for degree in [1,2,4,6,8]:
        matches=[x['raw'] for x in rows if x['mode']==mode and x['degree']==degree]
        ips=statistics.mean(x['ImagesPerSecond'] for x in matches)
        latencies=[v['Ms'] for x in matches for v in x['Rows']]
        sorted_ms=sorted(latencies)
        groups[mode][degree]=dict(ips=ips,mean_ms=statistics.mean(latencies),p95_ms=sorted_ms[int((len(sorted_ms)-1)*.95)],
            cpu_cores=statistics.mean(x['CpuCores'] for x in matches),
            rounds=[dict(ips=x['ImagesPerSecond'],mean_ms=x['MeanMs'],cpu_cores=x['CpuCores']) for x in matches])
peaks={m:max(values.items(),key=lambda kv:kv[1]['ips']) for m,values in groups.items()}
confirmation={}
for mode in ['A','B','C']:
    confirmation[mode]={}
    for degree in [1,6]:
        name=f'confirm-{mode}-c{degree}'
        r=json.loads((root/f'{name}.json').read_text(encoding='utf-8'))
        assert r['seconds']==30 and r['degree']==degree and r['Mismatches']==0 and not r['profile']
        assert r['WarmupSeconds']==8 and not r['GpuCls'] and not r['cpu'] and r['decode']
        assert not re.search(r'falling back|ErrorDeviceLost',(root/f'{name}.log').read_text(encoding='utf-8'),re.I)
        assert r['BinarySha256']==rows[0]['raw']['BinarySha256']
        assert r['Completed']==len(r['Rows']) and all(x['Match'] and x['Done']<=30 for x in r['Rows'])
        assert not any('alive=False' in state for state in r['states'])
        for image,result in zip(r['Images'],r['Baselines']):
            assert reference[mode,image['File']]==result
        confirmation[mode][degree]={k:r[k] for k in ['Completed','ImagesPerSecond','MeanMs','P95Ms','CpuCores','CpuMsPerImage']}
corpus={}
for mode in ['B','C']:
    doc=json.loads((root/f'corpus-{mode}.json').read_text(encoding='utf-8'))
    corpus[mode]={r['File']:r for r in doc['Rows']}
    assert len(corpus[mode])==90 and all(r['Success'] and r['Stable'] for r in doc['Rows'])
    assert not any('alive=False' in state for state in doc['States'])
assert corpus['B'].keys()==corpus['C'].keys()
assert all(corpus['B'][f]['Result']==corpus['C'][f]['Result'] for f in corpus['B'])
summary=dict(confirmation=confirmation,corpus_b_c_equal=90,groups=groups,peaks=peaks,total_completed=sum(x['raw']['Completed'] for x in rows),
             binary_sha256=rows[0]['raw']['BinarySha256'],b_c_full_results_equal=True)
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 并发扫描复核','','| 并发 | A 张/秒 | B 张/秒 | C 张/秒 | A 延迟ms | B 延迟ms | C 延迟ms |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for d in [1,2,4,6,8]:
    values=[groups[m][d] for m in ['A','B','C']]
    lines.append(f"| {d} | "+' | '.join(f"{v['ips']:.2f}" for v in values)+' | '+' | '.join(f"{v['mean_ms']:.2f}" for v in values)+' |')
(root/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
print('scanned peaks:',[(m,d,round(v['ips'],2),round(v['cpu_cores'],2)) for m,(d,v) in peaks.items()])

print('30-second confirmations:')
for mode in ['A','B','C']:
    for degree,r in confirmation[mode].items():
        print(mode,degree,round(r['ImagesPerSecond'],2),round(r['MeanMs'],2),round(r['CpuCores'],2))
