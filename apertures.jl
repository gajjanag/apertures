using FFTW
using Plots
using PGFPlots # MUST do this to avoid world age issue!
using Roots
using Optim
using Distributions
using Random
using LinearAlgebra
using ProgressMeter
using Primes
using LaTeXStrings

"""
Nazarov's ``magic'' Phi function x*arctan(x)-(1/2)log(1+x*x)
"""
function nazarov_phi(x::Real)
    return x*atan(x) - 0.5*log(1+x*x)
end

"""
Derivative Phi'(x)
"""
function nazarov_phiprime(x::Real)
    return atan(x)
end

"""
Nazarov's I(f)
"""
function nazarov_potential(fv::Array)
    s = 0
    for x in fv
        s += nazarov_phi(x)
    end
    return s
end

"""
Real orthonormal basis for the DFT. We only return non-DC terms.
"""
function dft_mat(n::Integer)
    mat = zeros(n,n)
    q = 2*pi/n
    h_pt = ceil(Int, (n+1)/2)
    # ceil((n+1)/2) cosines, floor((n-1)/2) sines
    for i=1:h_pt
        for j=1:n
            mat[i,j] = sqrt(2)*cos(q*(i-1)*(j-1))
        end
    end
    for i=h_pt+1:n
        for j=1:n
            mat[i,j] = sqrt(2)*sin(q*(i-h_pt)*(j-1))
        end
    end
    mat[1,:] /= sqrt(2)
    if iseven(n)
        mat[h_pt,:] /= sqrt(2)
    end
    return mat
end

function nazarov_psi(n::Integer)
    return dft_mat(n)[2:end,:]
end

"""
Compute the Nazarov constant for the DFT. In the paper's notation, this is M(n).
The paper gives the asymptotics, here one can numerically study it.
"""
function nazarov_const(n::Integer)
    mat = dft_mat(n)
    nc = 3*pi/2
    max_rat = 0
    for i=1:n
        nrat = 1/norm(mat[i,:],1)
        if (nrat > max_rat)
            max_rat = nrat
        end
    end
    max_rat *= n
    return nc*max_rat*max_rat
end

function plot_nazarov_const(n1::Integer, n2::Integer)
    x = primes(n1,n2)
    plot(x, [nazarov_const(i) for i in x])
end

"""
Some functions for checking the l1 norm beta(n) for the DFT. Asymptotics are given
in the paper. Finite n behavior can be studied with these routines, basically
the above nazarov_const is governed by the minimal value of the l1 norm over the
divisors of n.
"""
function l1_cos(n::Integer)
    sum = 0
    omega = 2*pi/n
    for i=0:n-1
        sum += abs(cos(omega*i))
    end
    sum /= n
    return sum-2/pi
end

function plot_l1_cos(n1::Integer, n2::Integer)
    x = n1:n2
    y = [l1_cos(n) for n in x]
    p = plot(x,y)
    savefig(p, "/tmp/l1_dft.pdf")
end

"""
Seed the greedy local optimum search randomly, works well in practice.
"""
function nazarov_seed_cortege(n::Integer)
    return rand([1,-1], n)
end

"""
Perform the local optimum search, given a vector that describes the power control.
"""
function nazarov_search!(a::Array, psi::Array, cortege::Array{Int})
    n = length(cortege)
    vn = length(psi[1,:])
    iter_order = [i for i in 1:n]
    fv = sum([cortege[j]*a[j]*psi[j,:] for j=1:n])
    cur_pot = nazarov_potential(fv)
    iter = 0
    eval_ct = 1
    is_opt = false
    println("-----------------------------------------------------------------")
    println("Performing local optimum search...")
    println("(iter, pot)")
    while !is_opt
        iter += 1
        print("($iter, $(cur_pot/vn))")
        randperm!(iter_order)
        is_opt = true
        for j in iter_order
            fv -= 2*cortege[j]*a[j]*psi[j,:]
            new_pot = nazarov_potential(fv)
            eval_ct += 1
            if new_pot > cur_pot
                is_opt = false
                cur_pot = new_pot
                cortege[j] = -cortege[j]
                break
            end
            fv += 2*cortege[j]*a[j]*psi[j,:]
        end
    end
    println("\n---------------------------------------------------------------")
    println("Function evaluations (scaled by n^2): $(eval_ct/(n*n))")
    println("\n---------------------------------------------------------------")
end

"""
Perform the shift/scale to get a [0,1] sequence from the locally optimal cortege.
We do not do the scaling of the paper, as it is conservative and meant for theorems.
The scaling here works better in practice and is natural.
"""
function _nazarov_aperture(a::Array, psi::Array, cortege::Array{Int})
    n = length(cortege)
    vn = length(psi[1,:])
    fv = sum([cortege[j]*a[j]*psi[j,:] for j=1:n])
    aper = zeros(vn)
    for i=1:vn
        aper[i] = nazarov_phiprime(fv[i])
    end
    # flip signs if positive
    if (sum(aper) > 0)
        aper = -aper
    end

    min_val = minimum(aper)
    max_val = maximum(aper)
    # scaling is trivial if nonnegative (very unlikely!)
    if (min_val > 0)
        warn("Something really weird, everything is nonnegative a priori!")
        aper /= max_val
        return aper
    end
    # the standard case. Note that this scaling is not the same used in the proof,
    # as the one used in the proof is a conservative one for theoretical soundness,
    # and we can improve upon it in code. Note that this scaling does not guarantee
    # \rho <= 0.5, though it is very close to it.
    for i=1:vn
        aper[i] = (aper[i]-min_val)/(max_val-min_val)
    end
    return aper
end

"""
General function that hooks things together: just takes in an n, a power control
vector (not necessarily normalized), and calls the above routines to get an aperture.
"""
function nazarov_aperture(n::Integer, powers::Array)
    psi = nazarov_psi(n)
    a = powers/norm(powers)
    cortege = nazarov_seed_cortege(n-1)
    nazarov_search!(a, psi, cortege)
    #redirect_stdout(() -> nazarov_search!(a, psi, cortege), open("/dev/null", "w"))
    return _nazarov_aperture(a, psi, cortege)
end

"""
Sanity checks on the generated aperture, and computation of its spectrum
"""
function nazarov_ahat(n::Integer, powers::Array, ctx::FFTW.cFFTWPlan)
    aperture = nazarov_aperture(n, powers)
    @assert(minimum(aperture) >= 0)
    @assert(maximum(aperture) <= 1)
    ahat = ctx * aperture
    ahat = [abs(x) for x in ahat]
    rho = ahat[1]/n
    println("rho (Nazarov): $rho")
    return ahat
end

"""
Scene correlation functions, generated by sampling at n equally spaced points
in [0,1] of a nonnegative, bounded, continuous function with symmetry about 1/2.
"""

"""
We first define the underlying d(x), for 0 <= x <= 0.5. All are underscored to
signify their "internal" nature.
"""
function _exp_corr(x::Real, theta::Real, beta::Real)
    return theta*(beta^x)
end

function _iid_corr(x::Real, theta::Real)
    return theta
end

function _char_corr(x::Real, theta::Real, beta::Real, gamma::Real)
    if (x < beta - gamma)
        return theta
    elseif (x > beta + gamma)
        return 0
    else
        return theta*(beta + gamma - x)/(2*gamma)
    end
end

function _power_corr(x::Real, theta::Real, beta::Real, gamma::Real)
    return theta/(1+beta*(x^gamma))
end

"""
General d vector generator based on sampling and symmetry about 1/2.
"""
function gen_corr(corr_fn, n::Integer)
    f = x -> (x < 0.5) ? corr_fn(x) : corr_fn(1-x)
    d = (1/n)*ones(n)
    for i=1:n
        j = i-1
        d[i] *= f(j/n)
    end
    return d
end

"""
Specializations to various cases of interest.
"""
function exp_corr(theta::Real, beta::Real, n::Integer)
    return gen_corr(x -> _exp_corr(x, theta, beta), n)
end

function iid_corr(theta::Real, n::Integer)
    return gen_corr(x -> _iid_corr(x, theta), n)
end

function char_corr(theta::Real, beta::Real, gamma::Real, n::Integer)
    return gen_corr(x -> _char_corr(x, theta, beta, gamma), n)
end

function power_corr(theta::Real, beta::Real, gamma::Real, n::Integer)
    return gen_corr(x -> _power_corr(x, theta, beta, gamma), n)
end

"""
Computation of power loss factor, i.e using a suboptimal rho. Here, a
(corresponding to W/J) is fixed. rhos describes an array of wherever we have
or consider spectrally flat constructions.
"""
function _power_loss_factor!(a::Real, rhos::Array)
    num = 2*a - 2*sqrt(a*(a+1))+1
    max_denom = 0
    for x in rhos
        denom = x*(1-x)/(a+x)
        if denom > max_denom
            max_denom = denom
        end
    end
    return num/max_denom
end

"""
Here we optimize (i.e maximize) _power_loss_factor over a, with an array of rhos
given (e.g rhos=[0.5, 0.25]). One may somewhat painfully do this rigorously;
this is just a sanity check.
"""
function power_loss_factor(rhos)
    # The 10 is completely ad-hoc, but works fine (see the plot)
    x = range(0, stop=10, length=10^8)
    y = zeros(length(x))
    i = 1
    @showprogress for i=1:length(x)
        y[i] = _power_loss_factor(x[i], rhos)
    end
    println("Loss factor: $(maximum(y))")
    x = range(0, stop=10, length=10^3)
    y = [_power_loss_factor(a, rhos) for a in x]
    p = plot(x, y)
    savefig(p, "/tmp/power_loss.pdf")
end

"""
LMMSE expressions.
"""
function eval_mmse(d::Array, n::Integer, t::Real, w::Real, j::Real, ahat::Array)
    mmse = 0
    rho = ahat[1]/n
    gamma = t/(n*(w+rho*j))
    for i=1:n
        mmse += 1/((1/d[i])+gamma*ahat[i]*ahat[i])
    end
    return mmse
end

function eval_mmse_flat(d::Array, rho::Real, n::Integer, t::Real, w::Real, j::Real)
    mmse = 0
    r = n*rho
    gamma = t/(n*(w+rho*j))
    mmse += 1/((1/d[1])+gamma*r*r)
    r = sqrt((n*n*rho*(1-rho))/(n-1))
    for i=2:n
        mmse += 1/((1/d[i])+gamma*r*r)
    end
    return mmse
end

function eval_mmse_rand(d::Array, rho::Real, n::Integer, t::Real, w::Real, j::Real, ctx::FFTW.cFFTWPlan)
    a = zeros(n)
    pd = Bernoulli(rho)
    ntr = round(Int, 400) # 400 is arbitrary, larger will make plot smoother
    mmse_arr = zeros(ntr)
    gamma = t/(n*(w+rho*j))
    for i=1:ntr
        rand!(pd, a)
        ahat = ctx * a
        mmse = 0
        for i=1:n
            r = abs(ahat[i])
            mmse += 1/((1/d[i])+gamma*r*r)
        end
        mmse_arr[i] = mmse
    end
    mmse = sum(mmse_arr)/ntr
    return mmse
end

function eval_mmse_lens(d::Array, n::Integer, t::Real, w::Real, j::Real)
    mmse = 0
    for i=1:n
        mmse += 1/((1/d[i])+((t*n)/(w+j)))
    end
    return mmse
end

function _mmse_power_sum(c::Real, d::Array, gamma::Real)
    n = length(d)
    p = 0
    for i=2:n
        p += max(0, (c-(1/d[i]))/gamma)
    end
    return p
end

function frac(x::Real)
    return x-floor(x)
end

"""
Waterfilling computation for power control.
"""
function waterfill_powers!(powers::Array, d::Array, rho::Real, n::Integer, t::Real, w::Real, j::Real)
    gamma = t/(n*(w+rho*j))
    # A hack for t=0
    if (gamma == 0)
        for i=1:n-1
            j = i+1
            powers[i] = 1/sqrt(n-1)
        end
        return
    end
    r = n*rho
    p = n*(floor(r)+frac(r)*frac(r))-r*r
    ub = (1/d[1])+gamma*p
    c = find_zero(x -> _mmse_power_sum(x,d,gamma)-p, (0,1.5*ub))
    for i=1:n-1
        j = i+1
        powers[i] = max(0, (c-(1/d[j]))/gamma)
    end
end

"""
Lower bound on LMMSE based on waterfilling.
"""
function mmse_lb(d::Array, rho::Real, n::Integer, t::Real, w::Real, j::Real)
    gamma = t/(n*(w+rho*j))
    # A hack for t=0
    if (gamma == 0)
        return sum(d)
    end
    r = n*rho
    p = n*(floor(r)+frac(r)*frac(r))-r*r
    ub = (1/d[1])+gamma*p
    c = find_zero(x -> _mmse_power_sum(x,d,gamma)-p, (0,1.5*ub))
    mmse = 0
    p = n*n*rho*rho
    mmse += 1/((1/d[1])+gamma*p)
    for i=2:n
        p = max(0, (c-(1/d[i])))
        mmse += 1/((1/d[i])+p)
    end
    return mmse
end

"""
Read the values of n for which we have spectrally flat constructions from
data/ into a dictionary of n->rho_arr[n].
"""
function get_spec_flat()
    fdict = Dict("data/1_2_vals.txt" => 0.5, "data/1_4_vals.txt" => 0.25, "data/1_8_vals.txt" => 0.125)
    spec_flat_dict = Dict{Int64,Array{Float64,1}}()
    for (file, rho) in fdict
        for line in eachline(file)
            n = parse(Int64, line)
            if !haskey(spec_flat_dict, n)
                spec_flat_dict[n] = [rho]
            else
                push!(spec_flat_dict[n], rho)
            end
        end
    end
    return spec_flat_dict
end

"""
Finally, the function that generates the plots of the paper, saved in "/tmp/mmse.pdf".
Specifically, set globally:
julia> n=677; theta=1; w=0.001; j=0.001; start=-50; stop=50; res=256;

For plot (a), do:
julia> beta = 1; gamma=0; plot_lmmse(n,theta,beta,gamma,w,j,start,stop,res);
For plot (b), do:
julia> beta = 0.02; gamma=0.005; plot_lmmse(n,theta,beta,gamma,w,j,start,stop,res);
"""
function plot_lmmse(n::Integer, theta::Real, beta::Real, gamma::Real, w::Real, j::Real, start::Real=0, len::Real=10, res=256)
    d = char_corr(theta, beta, gamma, n)
    fft_ctx = plan_fft(zeros(n))
    waterfill = zeros(n-1)
    x = range(start, length=res, stop=len)
    y_lb = zeros(length(x))
    popt = zeros(length(x))
    y_prng = zeros(length(x))
    y_spec = zeros(length(x))
    y_nazarov = zeros(length(x))
    spec_flat_dict = get_spec_flat()
    if !haskey(spec_flat_dict, n)
        error("No spectrally flat construction available at n=$n")
    else
        rho_arr = spec_flat_dict[n]
    end
    i = 1
    @showprogress for t in x
        t /= 10
        t = 10^t
        t *= n
        # lower bound
        res = Optim.optimize(rho->mmse_lb(d,rho,n,t,w,j), 0, 1)
        y_lb[i] = 10*log10(Optim.minimum(res))
        popt[i] = Optim.minimizer(res)
        waterfill_powers!(waterfill,d,popt[i],n,t,w,j)
        # random sequence
        res = Optim.optimize(rho->eval_mmse_rand(d,rho,n,t,w,j,fft_ctx),0,1)
        y_prng[i] = 10*log10(Optim.minimum(res))
        # spectrally flat, rho in rho_arr
        y_spec[i] = 10*log10(minimum([eval_mmse_flat(d,rho,n,t,w,j) for rho in rho_arr]))
        # Nazarov construction
        ahat_nazarov = nazarov_ahat(n, waterfill, fft_ctx)
        y_nazarov[i] = 10*log10(eval_mmse(d,n,t,w,j,ahat_nazarov))
        i += 1
    end
    # force pgfplots, GR has issues with LaTeX labels
    pgfplots()
    xlabel_str = L"10\log_{10}(t/n)"
    p = Plots.plot(x, [y_lb y_prng y_spec y_nazarov],
             label=["lower bound" "optimal random on-off" "spectrally flat" "Nazarov"],
             xlabel=xlabel_str,
             ylabel="LMMSE (dB)")
    plot_str = "/tmp/mmse.tex"
    savefig(p, plot_str)
    q = Plots.plot(x, popt,
             title="optimal p vs time for iid scene at\n (n,theta,w,j)=($n,$theta,$w,$j)",
             label="optimal p",
             xlabel=xlabel_str,
             ylabel="p")
    plot_str = "/tmp/optp_iid.tex"
    savefig(q, plot_str)
end
