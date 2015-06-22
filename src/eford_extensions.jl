
## Greate a GMM with only one mixture and initialize it to ML parameters _based on weighted data_
function GMM{T<:FloatingPoint}(x::DataOrMatrix{T}, w::Vector{T}; kind=:diag)
    sw, sx, sxx = stats(x, w, kind=kind)
    μ = sx' ./ sw                        # make this a row vector
    d = length(μ)
    if kind == :diag
        Σ = (sxx' - n*μ.*μ) ./ sw  # switched denom from n-1 to sw, since weights might all be small
    elseif kind == :full
        ci = cholinv((sxx - n*(μ'*μ)) / sw) # switched denom from n-1 to sw, since weights might all be small
        Σ = typeof(ci)[ci]
    else
        error("Unknown kind")
    end
    hist = History(@sprintf("Initlialized single Gaussian d=%d kind=%s with %d data points and weights",
                            d, kind, n))
    GMM(ones(T,1), μ, Σ, [hist], n)
end


function em_aist!(gmm::GMM, x::DataOrMatrix, sample_weights::Matrix ; nIter::Int = 10, varfloor::Float64=1e-3, sparse=0)
    size(x,2)==gmm.d || error("Inconsistent size gmm and x")
    d = gmm.d                   # dim
    ng = gmm.n                  # n gaussians
    nx = size(x,1)              # n data points
    initc = gmm.Σ
    ll = zeros(nIter)
    gmmkind = kind(gmm)
    for i=1:nIter
        ## E-step

        nₓ, ll[i], N, F, S = stats(gmm, x, parallel=true)
        ## M-step
        gmm.w = N / nₓ
        gmm.μ = F ./ N
        if gmmkind == :diag
            gmm.Σ = S ./ N - gmm.μ.^2
            ## var flooring
            tooSmall = any(gmm.Σ .< varfloor, 2)
            if (any(tooSmall))
                ind = find(tooSmall)
                warn("Variances had to be floored ", join(ind, " "))
                gmm.Σ[ind,:] = initc[ind,:]
            end
        elseif gmmkind == :full
            for k=1:ng
                if N[k] < d
                    warn(@sprintf("Too low occupancy count %3.1f for Gausian %d", N[k], k))
                else
                    μk = gmm.μ[k,:]
                    gmm.Σ[k] = cholinv(S[k] / N[k] - μk' * μk)
                end
            end
        else
            error("Unknown kind")
        end
        addhist!(gmm, @sprintf("iteration %d, average log likelihood %f",
                               i, ll[i] / (nₓ*d)))
    end
    if nIter>0
        ll /= nₓ * d
        finalll = ll[nIter]
    else
        finalll = avll(gmm, x)
        nₓ = size(x,1)
    end
    gmm.nx = nₓ
    addhist!(gmm,@sprintf("EM with %d data points %d iterations avll %f\n%3.1f data points per parameter",nₓ,nIter,finalll,nₓ/nparams(gmm)))
    ll
end


