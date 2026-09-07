# Measured qualification

Fugacio distinguishes numerical convergence, physical acceptance, and agreement
with experimental measurements. Each answers a different question. A small
solver residual doesn't establish phase stability, and a stable equilibrium
doesn't establish that its parameters represent a measured system accurately.

## Reproduce the complete workflow

```bash
uv sync --locked --all-packages
uv run python scripts/qualify.py --output artifacts/qualification
```

This command verifies the raw-data checksums, reconstructs the corpus, fits
ethanol-water using the 2011 publication, holds out the separate 2012 publication,
and builds an NRTL property package from the fit. It then heats a 10 mol/s,
equimolar feed from 330 K to 335 K at 20 kPa and independently audits the streams
and heater balance. The reference result needs approximately 2.820 kW.

The command writes a portable fit, a JSON report, and a Markdown coverage matrix.
CI runs it on every pull request and uploads these artifacts, including failed
and unsupported cases. Required holdout and process checks must pass. The
previously qualified cases in `benchmarks/thermodynamic-qualification.json` must
also remain qualified. Improving coverage requires new evidence; changing a
baseline doesn't change the physical error limits.

The first fit's independent holdout has approximately 3.71% pressure RMSE and
0.0348 vapor mole-fraction RMSE. Its fixed limits are 5% and 0.05, respectively.
These checks cover the published binary VLE observations. They don't establish
accurate excess enthalpy or global phase stability for ethanol-water.

## Corpus and ingestion

The package vendors seven raw XML records from the
[NIST ThermoML Archive](https://www.nist.gov/mml/acmd/trc/thermoml/thermoml-archive).
The source manifest retains each DOI, citation, retrieval date, URL, and SHA-256
digest, along with stable InChIKey-to-CAS identity mappings. The parser doesn't
identify components from a best-effort name match. Runtime and CI don't download
data or depend on an external chemical-name service.

The corpus has 447 experimental conditions across 20 binary systems: 283 VLE,
104 excess-enthalpy, and 60 cloud-point conditions. It includes near-ideal
aromatics, nonideal ethanol-water, associating alcohol mixtures, and partially
miscible methanol/hydrocarbon systems. Measurements and controlled conditions
retain their roles, phases, units, and reported uncertainties.

Complementary liquid/vapor tables join by identical independent conditions,
including component and phase identity. Row order isn't a join key. Eight source
rows with ambiguous rounded pressure/temperature keys are explicitly excluded
in the manifest. Every unselected source table also has an exclusion reason.
Unsupported quantities, units, transformed values, missing cells, conflicting
identifiers, and ambiguous joins raise errors.

Expanded uncertainty is converted to standard uncertainty only when the source
supplies a coverage factor. A 95% confidence statement alone doesn't imply a
factor of two. Missing uncertainties remain unknown. SI conversion scales
uncertainties without applying temperature offsets. Cloud-point observations
remain cloud points; they aren't fabricated liquid-liquid tie lines.

The archive supplies these records under its published terms and publisher
arrangements. Original citations and raw bytes are retained. Fugacio's software
license doesn't replace the source records' terms. The older sample XML files
and `ParameterBank.load_bundled()` remain synthetic demonstration fixtures,
separate from this measured corpus.

## Regression and holdout validation

```python
from fugacio.thermo.experimental import grouped_split, load_corpus
from fugacio.thermo.measured_regression import fit_measured_nrtl, validate_fit

data = tuple(o for o in load_corpus()
             if set(o.components) == {"ethanol", "water"})
train, test = grouped_split(
    data, by="source", holdout=("10.1016/j.fluid.2012.12.014",)
)
fit = fit_measured_nrtl(train)
validation = validate_fit(fit, test)
assert fit.diagnostics.converged and validation["accepted"]
fit.save("ethanol-water-fit.json")
```

NRTL fits four coefficients using `tau = a + b/T`, with fixed alpha. The internal
coordinates center the interaction at the mean training temperature. VLE and
excess enthalpy can be fitted together; the latter comes from the same model's
Gibbs-Helmholtz derivative. The default reference is PR saturation pressure with
ideal vapor and no Poynting or saturation-fugacity correction. Changing those
assumptions requires a new regression workflow.

Standard uncertainty weights apply where available for the fitted output.
Otherwise, `FitWeights` supplies explicit modeling scales: 2% pressure, 0.02
vapor mole fraction, and 100 J/mol excess enthalpy. Measured temperature and
liquid composition are treated as inputs. Their uncertainties are retained but
aren't silently converted into pressure uncertainties. This is weighted least
squares, not a full errors-in-variables likelihood.

Diagnostics include optimizer termination, Jacobian rank and condition number,
degrees of freedom, and local covariance in the centered parameter coordinates.
Rank-deficient fits fail even at a stationary objective. One isothermal VLE
table can't identify four independent temperature coefficients. Covariance is
conditional on the residual weights and exact inputs; it isn't a model-error
bound or a guarantee that the optimizer found the global best fit.

Splits accept explicit source DOIs, temperature keys, or source/dataset IDs.
Joined observations stay together. Missing split keys, empty partitions, and
training/validation ID overlap raise. Source holdouts provide stronger
independence than splitting adjacent temperatures in one experiment.

## Provenance and applicability

```python
from fugacio.sim import package_for
from fugacio.thermo.measured_regression import MeasuredFit

fit = MeasuredFit.load("examples/data/ethanol-water-fit.json")
pkg = package_for(fit.components, "nrtl", measured_fit=fit)
print(pkg.evidence.to_dict())
```

Package evidence is immutable JAX metadata. Parameter arrays remain
differentiable. Evidence distinguishes curated, predictive, measured-fit,
user-supplied, missing, and explicitly assumed-zero interactions. Cubic zero-kij
mixing assumptions are visible too. Provenance describes the constructed
package; if application code replaces parameter arrays, it must also update the
evidence rather than retaining the original fit's attribution.

`package_for(..., "nrtl")` and UNIQUAC reject missing pairs by default. To choose
zero interactions explicitly, pass `parameter_policy="allow_ideal"`. Existing
explicit `strict=False` calls remain an opt-in. UNIQUAC still includes its
combinatorial term when energetic interactions are zero. Predictive UNIFAC and
Dortmund are explicit method choices, never silent fallbacks.

Measured fits carry observed temperature, pressure, and composition bounds.
Checked calculations reject extrapolation by default. An explicit
`AcceptancePolicy(allow_extrapolation=True)` permits it while retaining failed
range flags in the report. Curated tables with no supplied validity interval
report unknown bounds. Unknown isn't empirically validated coverage. Pure-fluid
constants and heat-capacity correlations retain their existing database
provenance; binary fit evidence doesn't qualify these additional properties.

## Physical acceptance and process boundaries

```python
import jax.numpy as jnp
from fugacio.thermo.acceptance import flash_pt_checked, require_accepted

result = flash_pt_checked(pkg, 330.0, 20000.0, jnp.array([0.5, 0.5]))
require_accepted(result.report)
print(result.report.to_dict())
```

`flash_pt_with_info` retains the actual cubic, gamma-phi, or PC-SAFT iteration
report. `flash_pt_checked` adds input validity, component balance, present-phase
normalization, equifugacity, applicability, and a tangent-plane search in both
liquid and vapor phases. Only present phases require normalization. Exactly
absent components remain outside the stability-search support.

The stability search starts from the feed, returned phases, and each component
enrichment. Acceptance requires stationary trials and no negative observed
tangent-plane distance. It isn't a proof of a global Gibbs minimum. A stalled
trial fails the checked calculation. Disabling stability is an explicit policy
choice and appears as unchecked in the result. Failed checked values retain
their primals for diagnosis and have nonfinite forward and reverse derivatives.

`assess_flash(..., target=..., prop="enthalpy")` also checks energy; entropy
targets use `prop="entropy"`. The existing PH/PS report now includes
equifugacity. Energy-balanced units independently verify their returned phase
inventories, equilibrium residual, and parameter bounds before passing an
outlet downstream. Full stability checks are available through `heater_checked`,
`valve_checked`, and `audit_stream` after the numerical solve.

`audit_balance` verifies component and enthalpy flow closure using explicit heat
and shaft work, positive into the fluid. `audit_flowsheet` requires declared
`BalanceBoundary` objects, since stream endpoints alone don't specify heat or
work. Empty streams have no physical equilibrium composition and fail a full
stream audit. Reactive boundaries require explicit component generation and a
formation-consistent enthalpy model.

The copilot exposes `measured_corpus`, `inspect_thermodynamic_evidence`,
`fit_measured_binary`, and `checked_flash`. Existing flash and heater tools use
physical checks, and flash-temperature optimization verifies its final state.
Rejected physical states produce structured error reports. Other annotated
tools explicitly say when physical acceptance or empirical qualification wasn't
checked.

## Independent implementation oracle

`just clapeyron-oracles` uses Julia 1.10.10, Clapeyron 0.6.25, and the committed
Julia manifest. The dedicated CI job always runs; it doesn't depend on Python
`juliacall` being importable. Twelve PC-SAFT pressure cases compare pure fluids
and a binary mixture at vapor, intermediate, and liquid-like densities.
Pure parameters are checked explicitly and both implementations use zero binary
corrections. Parameter-bank differences therefore can't masquerade as kernel
errors. The relative pressure tolerance is 2e-5. This oracle covers the stated
nonassociating pressure kernels, while other existing oracle suites cover their
own properties and models.
