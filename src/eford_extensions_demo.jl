include(joinpath(Pkg.dir("GaussianMixtures"),"src","eford_extensions.jl"))

num_data = 1000
num_dim = 2
mu = [0.0, 0.0]
Sigma = diagm(ones(num_dim))
data = rand(MvNormal(mu,Sigma),num_data)'
data_weights = ones(num_data)
gmm = GMM(data, data_weights, kind=:full)
gmm0 = copy(gmm)

mu = [1.0, 0.0]
Sigma = diagm(ones(num_dim))
data = rand(MvNormal(mu,Sigma),num_data)'
em!(gmm,data,nIter=1)
gmm_em_1 = copy(gmm)
gmm = copy(gmm0)
em_aist!(gmm, data, data_weights, nIter = 1)
em_t2 = copy(gmm)



## stats: compute nth order stats for array _and weights_ (this belongs in stats.jl)
function stats{T<:FloatingPoint}(x::Matrix{T}, w::Vector{T}, order::Int=2; kind=:diag, dim=1)
    @assert( size(x,dim)==length(w) )
    n, d = nthperm([size(x)...], dim) ## swap or not trick
    if kind == :diag
        if order == 2
            return sum(w), vec(NumericExtensions.wsum(w, x, dim)), vec(NumericExtensions.wsumsq(w, x, dim))   # NumericExtensions is fast
        elseif order == 1
            return sum(w), vec(NumericExtensions.wsum(w, x, dim))
        else
            sw = sum(w)
            sx = zeros(T, order, d)
            for j=1:d
                for i=1:n
                    if dim==1
                        xi = xp = x[i,j]
                    else
                        xi = xp = x[j,i]
                    end
                    sx[1,j] += w[i]*xp
                    for o=2:order
                        xp *= xi
                        sx[o,j] += w[i]*xp
                    end
                end
            end
            return tuple([sw, map(i->vec(sx[i,:])/sw, 1:order)...]...)
        end
    elseif kind == :full
        order == 2 || error("Can only do covar starts for order=2")
        ## lazy implementation
        sw = sum(w)
        sx = vec(NumericExtensions.wsum(w, x, dim))
        # sxx = w x' * x
        sxx = vec(NumericExtensions.wsumsq(w, x, dim))
        error("Don't do this.  The code in the line above is wrong!  ")
        return sw, sx, sxx
    else
        error("Unknown kind")
    end
end

eval(:u_from_dist_t)([1.0, 2.0], 5.0)




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
        Δ = calc_distances(gmm, x)  ## Δ = (x_i - μ_k)' Λ_κ (x_i - m_k)  # nx x d
        # Compute ownership probabilities
        for k=1:ng # compute probability for point given that component
          # Σ's now are inverse choleski's, so logdet becomes -2sum(log(diag))
          normalization_k = 0.5d*log(2π) - sum(log(diag((gmm.Σ[k]))))  # WARNING: Assumes Gaussian
          epsilon[:,k] = -0.5*sumsq(Δ,2) .- normalization_k            # WARNING: Assumes Gaussian
        end
        logp = epsilon .+ log(gmm.w')  # weight by mixture component weight
        logsump = logsumexp(logp, 2) # normalization to get prob point from component
        broadcast!(-, logp, logp, logsump) # compute log prob point is from component
        epsilon = exp(logp)  # exp for prob point is from component
        ## E-step
        gmm.w = wsum(epsilon,prob_from_dist_func,Δ,1)
        ## M-step
        num = zeros(RT, ng, d); denom = zeros(RT, ng, d)
        for n in 1:nₓ
          u = u_from_dist(delta[n,:]) # nₓ x ng -> ng
          num .+= sample_weight[n]*(epsilon[n,:].*u)*x[n,:]

d = 2
ng = 4
nₓ = 10
sample_weight = randn(nₓ)
epsilon = randn(nₓ,ng)
x = randn(nₓ,d)

n = 1
num = sample_weight[n]*(epsilon[n,:]'.*u).*(x[n,:])
denom = vec(sample_weight[n]*(epsilon[n,:]'.*u))
num ./ denom

for n in 1:nₓ
  u = randn(ng)
  num += sample_weight[n]*(epsilon[n,:]'.*u).*(x[n,:])
  denom += vec(sample_weight[n]*(epsilon[n,:]'.*u))
end
num ./ denom

u = randn(nₓ,ng)
num = [zeros(d,d) for i in 1:ng]
denom = zeros(ng)
#vec(sample_weight[n]*(epsilon[n,:]'.*u).*(x[n,:]).*num)
for n in 1:nₓ
  C = broadcast(*, x[n,:], x[n,:]')
  num += map( x->x*C, (sample_weight[n]*epsilon[n,:]').*u[n,:]' )
  denom += sample_weight[n]*epsilon[n,:]'
end
num[1] / denom[1]

num
denom
num./denom

gamma(15)


size(x)
nthperm([size(x)...], 1)
[size(x)...]

Δ = Array(Float64, nₓ, ng)
epsilon = Array(Float64, nₓ, ng)
n=1
normalization = randn(ng)
epsilon[n,:].*exp(-0.5*Δ[n,:] .-normalization')
epsilon[n,:]
exp(-0.5*Δ[n,:] .-normalization')
Δ[n,:]
vec(Δ[n,:])
-normalization

