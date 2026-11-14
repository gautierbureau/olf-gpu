# Fast DC security analysis: a contingency that strands the slack bus loses the whole grid — upstream proposal

Status: draft for an upstream discussion (powsybl-open-loadflow), written 2026-11-14 from the topology-study benchmark on
`network.xiidm` (see `docs/tasks.md` #95). Two independent points:

- **A. Correctness (fast DC only).** When a contingency leaves the slack bus in a component smaller than the rest of the
  grid, the fast DC security analysis (`WoodburyDcSecurityAnalysis`) keeps the slack's component as the network and
  declares everything else lost. The AC and the plain DC security analyses relocate the slack to the largest component
  in that case. Fast and plain DC give different answers.
- **B. Robustness (any analysis).** `MostMeshedSlackBusSelector` counts every branch connected at both sides,
  retained switches included. When an analysis retains switches (operator strategies with switch actions, automation
  systems, switch contingencies), the slack moves to the bus bar with the most retained switches, which can sit in a
  weakly connected corner. The same network gets a different slack depending on which switches the study operates.

B is what exposed A on `network.xiidm`, but A stands on its own: any slack in a corner that one contingency can
strand triggers it.

## What was observed (measured)

`network.xiidm` (RTE, 2026-04-06): ~13,000 buses in the DC security analysis' network, ~46,000 when every breaker is
retained. RTE load flow parameters switched to DC; `slackBusSelectionMode` defaults to `MOST_MESHED`. The study
declares switch actions near 200 line contingencies (bus bar moves, coupler and breaker switching), so those switches
are retained.

- `MOST_MESHED` selects `AAHR_0`. It is one bus bar of the 400 kV substation AAHR whose 40 branches are nearly all
  retained switches. Its only external lines (LJJG, LJTU) lead into a corner that the line `LIYX` (DOA-ACLC) separates
  from the rest of the grid.
- Contingency `LIYX`, same operator strategy, three arms:

| arm | what `LIYX` disconnects | flow on line LIVC |
|---|---|---|
| DC security analysis (plain) | 263 MW of generation: the AAHR corner | −833.1 MW |
| fast DC security analysis, stock (KLU) | 58,059 MW of generation, 58,036 MW of load | not reported |
| fast DC on the GPU (mirrors the stock) | 58,059 MW of generation, 58,036 MW of load | not reported |

- With fast DC, the slack's component after `LIYX` is 15 buses (17 with every breaker retained). The other 13,347 buses
  (45,675) are "lost", so no branch result and no violation is reported for the rest of the country.
- The same contingency is also the costly one. Every operator strategy of `LIYX` rebuilds that lost part in the
  connectivity check: 12.1 s of a 12.5 s strategy phase (48,449 strategies, every breaker retained).
- With `slackBusSelectionMode = LARGEST_GENERATOR`, `LIYX` loses its real corner (42 buses; 61 with every breaker) in fast
  DC too, and the strategy phase takes 6.4 s.

## Mechanism (exact code)

**A.** `PropagatedContingency.findBusesAndBranchesImpactedBecauseOfConnectivityLoss` relocates the slack when
`relocateSlackBus` is set and the slack's component is not the largest (`isSlackBusIsolated`). It does this by excluding
the other buses from slack selection and making the slack's new component the main one:

```java
if (relocateSlackBus && isSlackBusIsolated(connectivity, network.getSynchronousNetworks().getFirst().getSlackBuses().getFirst())) {
    LOGGER.warn("Contingency '{}' leads to an isolated slack bus: relocate slack bus inside main component", contingencyId);
    Set<LfBus> excludedBuses = Sets.difference(Set.copyOf(network.getBuses()), connectivity.getLargestConnectedComponent());
    network.getSynchronousNetworks().forEach(lfScNetwork -> lfScNetwork.setExcludedSlackBuses(excludedBuses));
    connectivity.setMainComponentVertex(network.getSynchronousNetworks().getFirst().getSlackBuses().getFirst());
}
```

- `toLfContingency(network)`, used by the AC and DC security analyses, passes `true`.
- The fast DC path passes `false`: `ConnectivityBreakAnalysis.ConnectivityAnalysisResult.toLfContingency` calls
  `propagatedContingency.toLfContingency(network, false, analysis)`.
- The fast DC connectivity analysis (`computeConnectivityAnalysisResult`) takes `getVerticesRemovedFromMainComponent()`
  with the slack's component as main.
- Introduced with the fast DC security analysis (#1072), kept by #1192.

**B.** `MostMeshedSlackBusSelector`:

```java
private static int getBranchCountConnectedAtBothSides(LfBus bus) {
    return (int) bus.getBranches().stream().filter(LfBranch::isConnectedAtBothSides).count();
}
```

A retained switch is an `LfSwitch` branch, so it counts. A bus bar with many retained couplers and disconnectors wins,
whatever its electrical meshing. Without retained switches (the bus view), nothing changes.

## Why fast DC can relocate cheaply

The fast DC engine already handles a broken network without changing its matrix:
- The branches breaking connectivity are kept in the Woodbury update ("elements to reconnect").
- The lost buses' targets are zeroed and their injections are excluded from the slack distribution.
- The lost part therefore hangs on the kept branches with no injection, carries no flow, and the main part is exact.

Making the **largest** component the main one when the slack's is smaller changes only which side is "lost":
- The slack stays the angle reference of the equation system, through the kept bridge.
- The flows of the main part are unchanged by that choice, because angles are only defined up to a constant.
- The stranded corner's injections are excluded and redistributed over the main part's participating generators, as
  for any lost part.

Bus angles are then relative to a bus in the lost part. That matters only for absolute angle results; voltage angle
limits read differences.

## Proposal

**A (fix, fast DC).** In `ConnectivityBreakAnalysis` (the connectivity criterion confirmation and
`computeConnectivityAnalysisResult`), use the same rule as `PropagatedContingency`. When a contingency (or a contingency
and an operator strategy) leaves the slack's component smaller than the largest, take the largest as the main component
(`connectivity.setMainComponentVertex` on one of its buses) without moving the slack in the equation system.

Test: a small network where a branch contingency strands the slack bus in a pocket. Compare the fast and plain DC
security analyses: the same disconnected elements and the same branch flows in the main part. Today, fast DC loses
everything outside the pocket.

**B (robustness, slack selection).** Make the most-meshed count independent of the retained switches. Two options:
- Count only non-switch branches over the bus's closed-switch cluster: the electrical node a bus view would have built.
  The slack is then the cluster's bus with the most non-switch branches.
- Simpler: do not count `LfSwitch` branches. With every breaker retained, a bus bar then counts zero and a bay node one.
  The cluster version is the faithful one.

Either way, nothing changes when no switch is retained.

**Workaround meanwhile.** A study that retains switches should set its slack: `slackBusSelectionMode = LARGEST_GENERATOR`,
or `NAME` with `slackBusesIds`.

## Open questions

- A, multiple slack buses (`maxSlackBusCount > 1`): the main component is the one holding most of them, or the largest
  one?
- A, `createdSynchronousComponents` and the connectivity result's reported lost power once the main component changes
  side (the stranded corner becomes the lost part).
- B, is a change of the default slack acceptable when switches are retained? Results move for every study that retains
  switches, and the current choice is an artifact of the retention.
