"""Regenerate the bundled binary-parameter bank from the bundled ThermoML samples.

Runs the batch regression driver (:func:`fugacio.thermo.parameter_bank.
fit_bundled_samples`) over every bundled sample and writes the resulting bank to
``packages/fugacio-thermo/src/fugacio/thermo/parameter_bank.json``. The samples
are synthetic (see ``scripts/gen_thermoml_samples.py``), so the fits recover the
generating parameters and the bank is exactly reproducible.

Run from the repo root:

    uv run python scripts/gen_parameter_bank.py
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from fugacio.thermo.parameter_bank import ParameterBank, fit_bundled_samples  # noqa: E402

OUT = Path("packages/fugacio-thermo/src/fugacio/thermo/parameter_bank.json")


def main() -> None:
    entries = fit_bundled_samples()
    bank = ParameterBank(entries)
    OUT.write_text(bank.to_json())
    for e in bank.entries:
        print(
            f"  {e.components[0]} / {e.components[1]}: b12={e.b12:.2f} K, "
            f"b21={e.b21:.2f} K, rmse={e.rmse:.2e} ({e.source})"
        )
    print(f"wrote {OUT}: {len(bank)} pairs")


if __name__ == "__main__":
    main()
