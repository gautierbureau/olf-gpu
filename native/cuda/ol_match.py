import re, sys, collections
cpu={}
for l in open(sys.argv[1], errors='replace'):
    if not l.startswith('CPU_OL_STATS') or 'nosolve' in l: continue
    m=re.match(r'CPU_OL_STATS ctg=(\S+) firstNr=(\d+) totalNr=(\d+) outer=(\d+) status=(\S+) loops=(\S*)', l.strip())
    loops={t.split(':')[0]:t.split(':')[1] for t in m.group(6).split(',') if t}
    passes=1+sum(int(v.split('/')[0]) for v in loops.values())
    cpu[m.group(1)]=(passes,int(m.group(2)),int(m.group(3)),m.group(5),m.group(6))
gpu={}; dup=collections.Counter()
for l in open(sys.argv[2], errors='replace'):
    if not l.startswith('GPU_OL_STATS'): continue
    m=re.match(r'GPU_OL_STATS ctg=(\S+) passes=(\d+) firstNr=(\d+) totalNr=(\d+) status=(\S+)', l.strip())
    dup[m.group(1)]+=1
    gpu[m.group(1)]=(int(m.group(2)),int(m.group(3)),int(m.group(4)),int(m.group(5)))
both=[c for c in cpu if c in gpu]
diff=[c for c in both if cpu[c][3]=='CONVERGED' and gpu[c][3]==0 and (cpu[c][0],cpu[c][1],cpu[c][2])!=gpu[c][:3]]
same=sum(1 for c in both if cpu[c][3]=='CONVERGED' and gpu[c][3]==0 and (cpu[c][0],cpu[c][1],cpu[c][2])==gpu[c][:3])
print(f"matched ids {len(both)}; converged on both and identical (passes, first, total): {same}; different: {len(diff)}")
print(f"re-solved by robust retry (2 records): {[c for c,v in dup.items() if v>1]}")
print(f"not device-converged: {[(c,gpu[c][3],cpu[c][3]) for c in both if gpu[c][3]!=0]}")
print("ctg | CPU passes/first/total | GPU passes/first/total | CPU loops (re-solves/iters)")
for c in sorted(diff, key=lambda c:(gpu[c][2]-cpu[c][2])):
    print(f"  {c:14s} {cpu[c][0]:3d}/{cpu[c][1]}/{cpu[c][2]:<4d} {gpu[c][0]:3d}/{gpu[c][1]}/{gpu[c][2]:<4d} {cpu[c][4]}")
