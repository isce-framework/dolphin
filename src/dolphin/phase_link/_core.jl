function emi(C)
    Gam = abs.(C)
    Gam_inv = inv(Gam)
    A = Gam_inv .* C


    σ = 0.99
    F = lu(A - σ * I)
    Fmap = LinearMap{ComplexF64}((y, x) -> ldiv!(y, F, x), 25, ismutating=true)

    x = ones(eltype(A), size(A, 1))
    # lam_, v = IterativeSolvers.powm!(A, x; inverse=true, shift=σ)
    # lam_, v = IterativeSolvers.powm!(Fmap, x; inverse=true, shift=σ)
    # lam_, v = IterativeSolvers.powm!(Fmap, x; inverse=true, shift=σ)
    # λ, x = invpowm!(Fmap, x)
    # λ, x = invpowm!(Fmap, x)
    λ, x = invpowm(Fmap, shift=σ)
    return λ, v
end
