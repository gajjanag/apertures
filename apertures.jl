using Random
using Distributions
using Printf
using Optim
using StatsBase
using Primes
using FFTW
using LinearAlgebra
using Plots
using Roots

function nazarov_phi(x::Real)
    return x*atan(x) - 0.5*log(1+x*x)
end

function nazarov_phiprime(x::Real)
    return atan(x)
end

function nazarov_potential(fv::Array)
    s = 0
    for x in fv
        s += nazarov_phi(x)
    end
    return s
end

function nazarov_psi(n::Integer)
    return dft_mat(n)[2:end,:]
end

function nazarov_seed_cortege(n::Integer)
    return rand([1,-1], n)
end

function nazarov_search!(a::Array, psi::Array, cortege::Array{Int})
    n = length(cortege)
    vn = length(psi[1,:])
    iter_order = [i for i in 1:n]
    fv = sum([cortege[j]*a[j]*psi[j,:] for j=1:n])
    cur_pot = nazarov_potential(fv)
    iter = 0
    is_opt = false
    println("iter: pot")
    while !is_opt
        iter += 1
        println("$iter: $(cur_pot/vn)")
        randperm!(iter_order)
        is_opt = true
        for j in iter_order
            fv -= 2*cortege[j]*a[j]*psi[j,:]
            new_pot = nazarov_potential(fv)
            if new_pot > cur_pot
                is_opt = false
                cur_pot = new_pot
                cortege[j] = -cortege[j]
                break
            end
            fv += 2*cortege[j]*a[j]*psi[j,:]
        end
    end
end

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
    # and we can improve upon it in code.
    for i=1:vn
        aper[i] = (aper[i]-min_val)/(max_val-min_val)
    end
    return aper
end

function nazarov_aperture(n::Integer)
    psi = nazarov_psi(n)
    a = (1/sqrt(n-1))*ones(n-1)
    cortege = nazarov_seed_cortege(n-1)
    nazarov_search!(a, psi, cortege)
    return _nazarov_aperture(a, psi, cortege)
end

function nazarov_ahat(n::Integer, ctx::FFTW.cFFTWPlan)
    aperture = nazarov_aperture(n)
    @assert(minimum(aperture) >= 0)
    @assert(maximum(aperture) <= 1)
    ahat = ctx * aperture
    ahat = [abs(x) for x in ahat]
    rho = ahat[1]/n
    min_spec = minimum(ahat)
    loss_factor = n*rho*(1-rho)/(min_spec*min_spec)
    println("loss factor (Nazarov): $loss_factor")
    println("rho (Nazarov): $rho")
    return ahat
end

function nazarov_test(n::Integer)
    aperture = nazarov_aperture(n)
    @assert(maximum(aperture) <= 1)
    @assert(minimum(aperture) >= 0)
    rho = sum(aperture)/n
    power = sum([x^2 for x in aperture])/n
    println("rho: $rho")
    println("power: $power")
    ahat = fft(aperture)
    spec = [abs(x)^2 for x in ahat]
    min_spec = minimum(spec)/n
    println("min spec: $min_spec")
    theory_min = rho*(1-rho)
    loss_factor = rho*(1-rho)/min_spec
    println("Nazarov loss: $loss_factor")
    return aperture
end

function iid_corr(theta::Real, n::Integer)
    return (theta/n)*ones(n)
end

# constant factor loss for iid scene
function _loss_factor(a::Real)
    num = 2*a - 2*sqrt(a*(a+1))+1
    rhos = [0.25, 0.5]
    denoms = [x*(1-x)/(a+x) for x in rhos]
    denom = maximum(denoms)
    return num/denom
end

function exp_corr(theta::Real, beta::Real, n::Integer)
    d = (theta/n)*ones(n)
    h = ceil(Int, (n+1)/2)
    for i=2:h
        d[i] *= beta^((i-1)/(h-1))
        d[n+2-i] = d[i]
    end
    return d
end

function power_corr(theta::Real, beta::Real, gamma::Real, n::Integer)
    d = (theta/n)*ones(n)
    h = ceil(Int, (n+1)/2)
    d[1] *= beta^(-gamma)
    for i=2:h
        d[i] *= (beta*i)^(-gamma)
        d[n+2-i] = d[i]
    end
    d /= sum(d)
    d *= beta
    return d
end

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
    ntr = round(Int, 400)
    mmse_arr = zeros(ntr)
    gamma = t/(n*(w+rho*j))
    for i=1:ntr
        rand!(pd, a)
        ahat = ctx * a
        mmse = 0
        for i=1:n
            r = abs(ahat[i])
            mmse += 1/((1/d[1])+gamma*r*r)
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

function mmse_lb(d::Array, rho::Real, n::Integer, t::Real, w::Real, j::Real)
    gamma = t/(n*(w+rho*j))
    r = n*rho
    p = n*(floor(r)+frac(r)*frac(r))-r*r
    ub = (1/d[1])+gamma*p
    c = find_zero(x -> _mmse_power_sum(x,d,gamma)-p, (0,1.5*ub))
    mmse = 0
    p = n*n*rho*rho
    mmse += 1/((1/d[1])+gamma*p)
    for i=2:n
        p = max(0, (c-(1/d[i]))/gamma)
        mmse += 1/((1/d[i])+gamma*p)
    end
    return mmse
end

function plot_lmmse(n::Integer, theta::Real, w::Real, j::Real, start::Real=0, len::Real=4)
    d = iid_corr(theta, n)
    fft_ctx = plan_fft(zeros(n))
    x = range(start, length=256, stop=len)
    y_lb = zeros(length(x))
    popt = zeros(length(x))
    y_prng = zeros(length(x))
    y_spec = zeros(length(x))
    y_nazarov = zeros(length(x))
    ahat_nazarov = nazarov_ahat(n, fft_ctx)
    rho_arr = [0.5]
    i = 1
    for t in x
        t *= n
        # lower bound
        res = optimize(rho->mmse_lb(d,rho,n,t,w,j), 0, 1)
        y_lb[i] = 10*log10(Optim.minimum(res))
        popt[i] = Optim.minimizer(res)
        # random sequence
        res = optimize(rho->eval_mmse_rand(d,rho,n,t,w,j,fft_ctx),0,1)
        y_prng[i] = 10*log10(Optim.minimum(res))
        # spectrally flat, rho=1/2,1/4,1/8
        y_spec[i] = 10*log10(minimum([eval_mmse_flat(d,rho,n,t,w,j) for rho in rho_arr]))
        # Nazarov construction (iid case only, FIXME)
        y_nazarov[i] = 10*log10(eval_mmse(d,n,t,w,j,ahat_nazarov))
        i += 1
    end
    p = plot(x, [y_lb, y_prng, y_spec, y_nazarov],
             title="LMMSE (dB) vs time for iid scene at\n (n,theta,w,j)=($n,$theta,$w,$j)",
             label=["lower bound", "opt rand p", "spec flat", "Nazarov"],
             xlabel="exposure time (divided by n)",
             ylabel="LMMSE")
    plot_str = "/tmp/mmse_iid.pdf"
    savefig(p, plot_str)
    q = plot(x, popt,
             title="optimal p vs time for iid scene at\n (n,theta,w,j)=($n,$theta,$w,$j)",
             label="optimal p",
             xlabel="exposure time (divided by n)",
             ylabel="p")
    plot_str = "/tmp/optp_iid.pdf"
    savefig(q, plot_str)
end

function auto_corr(v::Array)
    n = length(v)
    cv = zeros(n)
    for i=0:n-1
        for j=0:n-1
            cv[i+1] += v[j+1]*v[((j+i)%n)+1]
        end
    end
    return cv
end

function countnz(v::Array)
    return count(i->i!=0, v)
end

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
    savefig(p, "/tmp/foo.pdf")
end

function plot_nazarov_const(n1::Integer, n2::Integer)
    x = primes(n1,n2)
    plot(x, [nazarov_const(i) for i in x])
end

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
