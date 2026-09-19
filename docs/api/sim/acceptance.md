# Process acceptance

Post-solve physical audits of process results: `audit_stream` for a stream's
phase state, `audit_balance` for component and energy closure at a boundary,
the checked unit wrappers (`flash_drum_checked`, `heater_checked`,
`valve_checked`), and `audit_flowsheet` over declared `BalanceBoundary`
objects. See the [measured qualification guide](../../qualification.md#physical-acceptance-and-process-boundaries)
for the criteria and their limits.

::: fugacio.sim.acceptance
