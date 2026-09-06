# Foreign implementation checks. Inputs and outputs are plain JSON; no Python
# bridge, lazy package installation, or optional skip occurs in this job.
using Clapeyron, JSON, LinearAlgebra

VERSION == v"1.10.10" || error("Expected Julia 1.10.10")
pkgversion(Clapeyron) == v"0.6.25" || error("Expected Clapeyron 0.6.25")
cases = JSON.parsefile(ARGS[1])
results = []
for case in cases
    n = length(case["components"])
    model = PCSAFT(case["components"])
    Clapeyron.set_k!(model, zeros(n, n))
    maximum(abs, Clapeyron.get_k(model)) <= 1e-12 || error("Expected zero binary correction")
    z = Float64.(case["composition"])
    T = Float64(case["temperature_k"])
    rho = Float64(case["density_mol_m3"])
    # The input includes the exact pure parameters so database drift cannot be
    # mistaken for a kernel mismatch. Fail if either bank changes.
    for (key, expected) in case["parameters"]
        actual = getproperty(model.params, Symbol(key)).values
        if key == "sigma"
            actual = diag(actual)
        elseif key == "epsilon"
            actual = diag(actual)
        end
        isapprox(vec(actual), Float64.(expected); rtol=1e-6) || error("Parameter mismatch: " * key)
    end
    p = pressure(model, 1/rho, T, z)
    push!(results, Dict("id" => case["id"], "pressure_pa" => p))
end
open(ARGS[2], "w") do io
    JSON.print(io, Dict("julia" => string(VERSION), "clapeyron" => string(pkgversion(Clapeyron)),
                        "results" => results), 2)
end
