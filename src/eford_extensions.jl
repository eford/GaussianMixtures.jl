## train.jl  Likelihood calculation and em training for T Distribution Mixture Model using weighted data
## (c) 2015 Eric B. Ford
## extends (and borrows heavily from) GaussianMixtures's package by David A. van Leeuwen)

using NumericExtensions
using Distributions
using GaussianMixtures

## Greate a GMM with only one mixture and initialize it to ML parameters _based on weighted data_
function GaussianMixtures.GMM{T<:FloatingPoint}(x::GaussianMixtures.DataOrMatrix{T}, w::Vector{T}; kind=:diag)
    n, d = size(x)
    @assert n == length(w)
    sw = sum(w)
    sx = vec(NumericExtensions.wsum(w, x, 1))
    μ = sx' ./ sw                        # make this a row vector
    if kind == :diag
        warn("I haven't bothered to write specializations for diagnoal matrices for weighted data.")
    end
    if kind == :full ||  kind == :diag
       covar = Array(T,d,d)
       for i in 1:n
         diff = (x[i,:].-μ)
         covar .+= w[i].*broadcast(*, diff, diff')
       end
       covar ./= sw
       ci = GaussianMixtures.cholinv(covar)
       Σ = typeof(ci)[ci]
    else
        error("Unknown kind")
    end
    hist = GaussianMixtures.History(@sprintf("Initlialized single Gaussian d=%d kind=%s with %d data points and weights giving effective sample size of %f",
                            d, kind, n, sw))
    GaussianMixtures.GMM(ones(T,1), μ, Σ, [hist], n)
end

## compute distance of each sample from each mixture location, output: nₓ × ng, based on llpg for full covar
function calc_distances{GT,T<:FloatingPoint}(gmm::GMM{GT,GaussianMixtures.FullCov{GT}}, x::Matrix{T})
    RT = promote_type(GT,T)
    (nₓ, d) = size(x)
    ng = gmm.n
    d==gmm.d || error ("Inconsistent size gmm and x")
    dist = Array(RT, nₓ, ng)
    Δ = Array(RT, nₓ, d)
    for k=1:ng
        ## Δ = (x_i - μ_k)' Λ_κ (x_i - m_k)
        GaussianMixtures.xμTΛxμ!(Δ, x, gmm.μ[k,:], gmm.Σ[k])
        dist[:,k] = sumsq(Δ,2)
    end
    return dist::Matrix{RT}
end

u_from_dist_gaussian(dist::Vector{Float64}, d::Float64) = ones(length(dist))  # I think this is right for a Gaussian, but I should check to be sure.
u_from_dist_t(dist::Vector{Float64}, d::Float64; nu::Float64= 4.0) = (nu+d)./(nu+dist)
#u_from_dist(dist::Vector{Float64}) = u_from_dist_gaussian(dist)

function em_aist!{GT,T<:FloatingPoint}(gmm::GMM{GT,GaussianMixtures.FullCov{GT}}, x::DataOrMatrix{T}, sample_weight::Vector{T} ; nIter::Int = 10, varfloor::Float64=1e-3, sparse=0, u_from_dist=:u_from_dist_t, nu::Float64 = 4.0)
    RT = promote_type(GT,T)
    size(x,2)==gmm.d || error("Inconsistent size gmm and x")
    d = gmm.d                   # dim
    ng = gmm.n                  # n gaussians
    nₓ = size(x,1)              # n data points
    initc = gmm.Σ
    ll = zeros(nIter)
    gmmkind = kind(gmm)
    Δ = Array(RT, nₓ, d)
    epsilon = Array(RT, nₓ, ng)
    for i=1:nIter
      # Compute distance's
      Δ = calc_distances(gmm, x)  ## Δ = (x_i - μ_k)' Λ_κ (x_i - m_k)  # nx x ng
      # Compute ownership probabilities
      normalization = Array(RT,ng)
      for k=1:ng # compute probability for point given that component
        # Σ's now are inverse choleski's, so logdet becomes -2sum(log(diag))
        if u_from_dist == :u_from_dist_gaussian
          normalization[k] = -0.5d*log(2π) + sum(log(diag((gmm.Σ[k]))))  # WARNING: Assumes Gaussian
          epsilon[:,k] = -0.5*Δ[:,k] .+ normalization[k]            # WARNING: Assumes Gaussian
        elseif u_from_dist == :u_from_dist_t
          normalization[k] = lgamma((nu+d)/2)-lgamma(nu/2)-0.5*d*log(pi*nu) + sum(log(diag((gmm.Σ[k]))))  # WARNING: Assumes t_nu
          epsilon[:,k] = -0.5*(nu+d)*log(1.0+Δ[:,k]) .+ normalization[k]            # WARNING: Assumes t_nu
        else
          println("# Don't know what PDF to use with u function.")
        end

      end
      logp = epsilon .+ log(gmm.w') # weight by mixture component weight
      logsump = logsumexp(logp, 2)  # normalization to get prob point from component
      broadcast!(-, logp, logp, logsump) # compute log prob point is from component
      epsilon = exp(logp)           # nₓ x ng: exp for prob point is from component
      n_eff_in_component = sum(epsilon,1)    # ng

      ## E-step
      #println("Pre: size(w)= ", size(gmm.w) )
      gmm.w = zeros(ng)
      #println("size(w)= ", size(gmm.w), " size(eps)= ",size(epsilon), " sizeof(dist) = ", size(Δ), " sizeof(norm)= ", size(normalization))
        if u_from_dist == :u_from_dist_gaussian
           gmm.w = vec(logsumexp(  log(epsilon).-0.5*Δ,1))
           gmm.w += normalization
        elseif u_from_dist == :u_from_dist_t
           gmm.w = vec(logsumexp(  log(epsilon).-0.5*(nu+d)*log(1.0+Δ),1))
           gmm.w += normalization
        else
          println("# Don't know what PDF to use with u function.")
        end
      #println("Post: size(w)= ", size(gmm.w) )
      gmm.w = exp(gmm.w)/exp(logsumexp(gmm.w))  # Do we need this to make sure weights add up?
      ## M-step
      num = zeros(RT, ng, d); denom = zeros(RT, ng)
      for n in 1:nₓ
        w_eps_u = sample_weight[n]* epsilon[n,:]'.* eval(u_from_dist)(vec(Δ[n,:]),convert(Float64,d),nu=nu) # ng
        num .+= w_eps_u .*x[n,:]    # ng x d
        denom .+= w_eps_u           # ng
      end
      gmm.μ = num ./ denom                  # ng x d
      if gmmkind == :diag
         warn(@sprintf("Should specialize for a diagonal covariance matrix"))
      end
      if gmmkind == :full ||  gmmkind == :diag  # For now use general version
        # Do we need to some sort of check to make sure there are enough points to estimate the covariance?
        for k in 1:ng
          if n_eff_in_component[k] < d
            warn(@sprintf("Too low effective occupancy count %3.1f for Component %d", n_eff_in_component[k], k))
          end
        end
        num = Array{RT, 2}[zeros(d,d) for i in 1:ng]
        fill!(denom, 0.0)
        for n in 1:nₓ
          u = eval(u_from_dist)(vec(Δ[n,:]),convert(Float64,d),nu=nu)             # ng
          w_eps = sample_weight[n]* epsilon[n,:]'  # ng
          denom .+= w_eps                          # ng
          for k in 1:ng
            diff = (x[n,:].-gmm.μ[k,:])
            C = broadcast(*, diff, diff')
            num[k] .+= w_eps[k].*u[k]*C          # ng x (d x d)
           end
        end
        for k in 1:ng
          local new_cholinv_Sigma_k
          try 
             new_cholinv_Sigma_k = GaussianMixtures.cholinv( num[k] / denom[k])                      # ng x (d x d)
          catch
             warn(string("# cholinv failed, so not updating covariance matrix for comonent ",k," with weight ", gmm.w[k]))
          end
          gmm.Σ[k] = new_cholinv_Sigma_k
        end
      else
        error("Unknown kind")
      end

      # Would have to figure out ll, avll and finalll to report these
      GaussianMixtures.addhist!(gmm, @sprintf("iteration %d", i)) #, " average log likelihood %f", i, ll[i] / (nₓ*d)))
    end
    #if nIter>0
    #    ll /= nₓ * d
    #    finalll = ll[nIter]
    #else
    #    finalll = avll(gmm, x)
    #    nₓ = size(x,1)
    #end
    gmm.nx = nₓ
    #GaussianMixtures.addhist!(gmm,@sprintf("EM with %d data points %d iterations avll %f\n%3.1f data points per parameter",nₓ,nIter,finalll,nₓ/nparams(gmm)))
    GaussianMixtures.addhist!(gmm,@sprintf("EM with %d data points %d iterations avll XXXX.\n%3.1f data points per parameter",nₓ,nIter,nₓ/nparams(gmm)))
    #ll
    gmm
end

