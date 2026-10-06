import sys, re, collections
# CPU_OL_STATS ctg=X firstNr=a totalNr=b outer=c status=S loops=Name:n/it,...
rows=[]
for line in open(sys.argv[1], errors='replace'):
    if not line.startswith('CPU_OL_STATS'): continue
    m=re.match(r'CPU_OL_STATS ctg=(\S+) (.*)', line.strip())
    if 'nosolve' in m.group(2): rows.append((m.group(1),None)); continue
    d=dict(kv.split('=',1) for kv in m.group(2).split())
    loops={}
    for t in d.get('loops','').split(','):
        if t: n,v=t.split(':'); a,b=v.split('/'); loops[n]=(int(a),int(b))
    rows.append((m.group(1),(int(d['firstNr']),int(d['totalNr']),int(d['outer']),d['status'],loops)))
solved=[r for r in rows if r[1]]
conv=[r for r in solved if r[1][3]=='CONVERGED']
print(f"contingencies {len(rows)}, solved {len(solved)}, converged {len(conv)}")
# resolves = number of outer-loop re-solves that actually ran Newton (each loop entry counts re-solves)
h=collections.Counter(); hf=collections.Counter(); ht=collections.Counter()
for _,(f,t,o,st,lp) in conv:
    k=sum(v[0] for v in lp.values()); h[k]+=1; hf[k]+=f; ht[k]+=t
tf=sum(hf.values()); tt=sum(ht.values())
print(f"Newton iterations: first inner solve {tf}, total {tt} -> outer-loop re-solves cost {tt-tf} ({100*(tt-tf)/tt:.1f}%)")
print("solve passes (1 + outer re-solves) | contingencies | first-NR iters | total-NR iters")
for k in sorted(h): print(f"  {k+1:3d} | {h[k]:6d} | {hf[k]:7d} | {ht[k]:7d}")
per=collections.defaultdict(lambda:[0,0,0])
for _,(f,t,o,st,lp) in conv:
    for n,(a,b) in lp.items(): per[n][0]+=1; per[n][1]+=a; per[n][2]+=b
print("outer loop | contingencies it re-solved | re-solves | NR iterations")
for n,(c,a,b) in sorted(per.items(), key=lambda x:-x[1][2]): print(f"  {n:34s} {c:6d} {a:7d} {b:7d}")
