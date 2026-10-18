# Behaviour switches

`docs/env-vars.md` catalogues everything readable from the environment. This file lists the switches that
**change what the device computes**, so that each one can be given a disposition. The rule it enforces: *once a
path matches the CPU better, DELETE the other.* A switch whose other mode is simply wrong is not a feature; it is
a measurement that was never acted on.

## State (2026-10-18): none left

The configuration cleanup removed every behaviour switch:

- every `-Dolf.gpu.*` system property;
- every environment variable that selected a solve path or a numeric mode, for example the batch mode, the
  reordering algorithm, the coupler-tree relocation, the hybrid and non-batch engines, and the pilot-self masks.

The cuDSS settings that stay are parameters, as in OLF: `GpuLoadFlowParameters` (config module
`open-loadflow-gpu-default-parameters`) holds the reordering, matching and pivoting algorithms, the pivot threshold
and the iterative-refinement steps.

The environment variables still read are traces: they print, and they do not change a result. Two of them have
names that suggest otherwise:

- `OLF_GPU_PSG_STEP` prints one pilot-self group's layout after each kernel of the reactive-limits pass.
- `OLF_GPU_CPU_ROUTED` names the contingencies the device hands back to the CPU; it does not route any.

## Adding one

A new switch needs the user's agreement first, and in a form OLF itself would use: a parameter, not an environment
variable. Before proposing one, measure both modes on the full branch scope and keep only the one that matches the
CPU.
