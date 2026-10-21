"""Official pypowsybl SA on the harness's exact scope: RTE JSON load-flow parameters, the harness's
increased-violations thresholds, threadCount from argv. Times run_ac only (network loading excluded, as the
harness's CPU timer does). Writes {ctg: [status, [[subject, type, side, name, value, limit], ...]]} to argv[4]."""
import json, sys, time
import pypowsybl as pp

case, ctg_file, threads, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
params_file = '/home/bureaugau/olf-gpu/powsybl-open-loadflow-gpu/rte-loadflow-default-parameters.json'
t0 = time.time()
net = pp.network.load(case)
print(f'network loaded in {time.time() - t0:.1f} s', flush=True)
lfp = pp.loadflow.Parameters.from_json(open(params_file).read())
# Java new IncreasedViolationsParameters(0.5, 0.001, 0.5, 0.001, 0.001) =
# (lowVoltageAbsolute, lowVoltageProportional, highVoltageAbsolute, highVoltageProportional, flowProportional)
ivp = pp.security.IncreasedViolationsParameters(flow_proportional_threshold=0.001,
                                                low_voltage_proportional_threshold=0.001,
                                                low_voltage_absolute_threshold=0.5,
                                                high_voltage_proportional_threshold=0.001,
                                                high_voltage_absolute_threshold=0.5)
sap = pp.security.Parameters(load_flow_parameters=lfp, increased_violations_parameters=ivp,
                             provider_parameters={'threadCount': threads})
sa = pp.security.create_analysis()
sa.add_contingencies_from_json_file(ctg_file)
t1 = time.time()
res = sa.run_ac(net, sap, provider='OpenLoadFlow')
wall = time.time() - t1
print(f'SA_WALL {wall * 1000:.0f} ms threads={threads}', flush=True)
statuses = {}
for cid, r in res.post_contingency_results.items():
    statuses[cid] = [r.status.name, []]
lv = res.limit_violations.reset_index()
for row in lv.itertuples(index=False):
    cid = row.contingency_id
    if cid in statuses:
        statuses[cid][1].append([row.subject_id, str(row.limit_type), str(row.side), str(row.limit_name),
                                 float(row.value), float(row.limit)])
from collections import Counter
print('statuses', dict(Counter(v[0] for v in statuses.values())))
print('violations', sum(len(v[1]) for v in statuses.values()))
json.dump({'wall_ms': wall * 1000, 'post': statuses}, open(out, 'w'))
