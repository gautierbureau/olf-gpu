"""pypowsybl, official: base load flow and a one-contingency SA on network.xiidm with the RTE parameters, 1 thread;
prints the report lines about the three oscillating buses and the PV<->PQ switches."""
import re, sys
import pypowsybl as pp
net = pp.network.load('/home/bureaugau/olf-gpu/powsybl-open-loadflow-gpu/network.xiidm')
lfp = pp.loadflow.Parameters.from_json(open('/home/bureaugau/olf-gpu/powsybl-open-loadflow-gpu/rte-loadflow-default-parameters.json').read())
rep = pp.report.ReportNode()
res = pp.loadflow.run_ac(net, lfp, report_node=rep)
print('BASE LF:', res[0].status, 'iterations', res[0].iteration_count)
txt = str(rep)
lines = [l for l in txt.splitlines() if re.search(r'IZTH_0|IZTH_2|IZSV_0|IZUM_0|PV -> PQ|PQ -> PV|Outer loop', l)]
print('BASE report lines:', len(lines)); print('\n'.join(lines[:40]))
ivp = pp.security.IncreasedViolationsParameters(flow_proportional_threshold=0.001, low_voltage_proportional_threshold=0.001,
                                                low_voltage_absolute_threshold=0.5, high_voltage_proportional_threshold=0.001,
                                                high_voltage_absolute_threshold=0.5)
sap = pp.security.Parameters(load_flow_parameters=lfp, increased_violations_parameters=ivp, provider_parameters={'threadCount': '1'})
sa = pp.security.create_analysis()
sa.add_contingencies_from_json_file(sys.argv[1])
rep2 = pp.report.ReportNode()
r = sa.run_ac(pp.network.load('/home/bureaugau/olf-gpu/powsybl-open-loadflow-gpu/network.xiidm'), sap, provider='OpenLoadFlow', report_node=rep2)
print('SA:', {k: v.status.name for k, v in r.post_contingency_results.items()})
txt2 = str(rep2)
i = txt2.find('LIRN')
sec = txt2[i:] if i >= 0 else txt2
lines2 = [l for l in sec.splitlines() if re.search(r'IZTH_0|IZTH_2|IZSV_0|IZUM_0|PV -> PQ|PQ -> PV|Outer loop|outer loop', l)]
print('CONTINGENCY report lines:', len(lines2)); print('\n'.join(lines2[:60]))
open(sys.argv[2], 'w').write(txt2)
