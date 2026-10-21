"""Official pypowsybl: a security analysis with ONE contingency on network.xiidm, RTE load-flow parameters, 1 thread.
OLF's report shows the ReactiveLimits oscillation of the three controllers of IZUM_0's remote voltage control
(IZTH_0, IZTH_2, IZSV_0): released PQ->PV on a ~1e-7 pu voltage deficit, pinned again, until maxPqPvSwitch.
usage: python pp_rl_oscillation.py [contingencyBranchId=LIRN] [network=network.xiidm] [params=rte-loadflow-default-parameters.json]"""
import re, sys
import pypowsybl as pp

ctg = sys.argv[1] if len(sys.argv) > 1 else 'LIRN'
network_file = sys.argv[2] if len(sys.argv) > 2 else 'network.xiidm'
params_file = sys.argv[3] if len(sys.argv) > 3 else 'rte-loadflow-default-parameters.json'

lfp = pp.loadflow.Parameters.from_json(open(params_file).read())
sap = pp.security.Parameters(load_flow_parameters=lfp, provider_parameters={'threadCount': '1'})
sa = pp.security.create_analysis()
sa.add_single_element_contingency(ctg, ctg)
report = pp.report.ReportNode()
result = sa.run_ac(pp.network.load(network_file), sap, provider='OpenLoadFlow', report_node=report)
print(f"contingency {ctg}: {result.post_contingency_results[ctg].status.name}")
text = str(report)
section = text[text.find(ctg):] if ctg in text else text
for line in section.splitlines():
    if re.search(r"Outer loop iteration|buses switched|blocked PQ|Switch bus '(IZTH_0|IZTH_2|IZSV_0)'", line):
        print(line.rstrip())
