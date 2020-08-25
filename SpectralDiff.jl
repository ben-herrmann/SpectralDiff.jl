module SpectralDiff

export fourier_grid, fourier_diff, fourier_weights
export cheb_grid, cheb_diff, cheb_weights
export her_grid, her_diff, her_weights

using LinearAlgebra

function fourier_grid(N::Integer)
    x = collect(0:N-1)*2π/N
    return x
end

function fourier_diff(N::Integer,degree::Integer=1)
    Δx=2π/N
    if degree%2≠0
        D1 = zeros(N,N)
        for i=1:N, j=1:N
            if i≠j
                D1[i,j] = 0.5*(-1)^(i+j)*cot(0.5*(i-j)*Δx)
            end
        end
    elseif degree>=2
        D2 = fill(-π^2/(3*Δx^2)-1/6,(N,N))
        for i=1:N, j=1:N
            if i≠j
                D2[i,j] = -0.5*(-1)^(i+j)*csc(0.5*(i-j)*Δx)^2
            end
        end
    end
    if degree==1
        D=D1
    elseif degree%2==0
        D=D2^(degree/2)
    else
        D = D2^div(degree,2)*D1^(degree%2)
    end
    return D
end

function fourier_weights(N::Integer)
    w = 2π/N*ones(N)
    return w
end

function cheb_grid(N::Integer)
    x = cos.(π.*(0:N-1)./(N-1))
    return x
end

function cheb_diff(N::Integer)
    x = cheb_grid(N)
    c = [2; ones(N-2); 2].*(-1).^(0:N-1)
    X = repeat(x,1,N)
    dX = X-X'
    D = (c*(1 ./c)')./(dX+I)
    D = D-Diagonal(vec(sum(D,dims=2)))
    return D
end

function cheb_weights(N::Integer)
    θ = π*(0:N-1)/(N-1)
    x = cos.(θ)
    w = zeros(N)
    v = ones(N-2)
    if N%2 ≠ 0
        w[1] = 1/((N-1)^2-1)
        w[N] = w[1]
        for k=1:(N-1)/2-1
            v = v.-2*cos.(2*k*θ[2:N-1])/(4*k^2-1)
        end
        v = v.-cos.((N-1)*θ[2:N-1])/((N-1)^2-1)
    else
        w[1] = 1/(N-1)^2
        w[N] = w[1]
        for k=1:(N-2)/2
            v = v.-2*cos.(2*k*θ[2:N-1])/(4*k^2-1)
        end
    end
    w[2:N-1] = 2*v/(N-1)
    return w
end

function her_grid(N::Integer,b::Real=1.0)
    #  J.A.C. Weideman, S.C. Reddy 1998.
    J = SymTridiagonal(zeros(N),sqrt.(1:N-1))
    r = eigvals(J)/sqrt(2)
    x = r/b
    return x
end

function her_diff(N::Integer,b::Real=1.0,degree::Integer=1.0)
    #  J.A.C. Weideman, S.C. Reddy 1998.
    x = her_grid(N)
    α = exp.(-x.^2/2)
    β = Array{typeof(x[1])}(undef,degree+1,N)
    β[1,:] = ones(length(x))
    β[2,:] = -x
    for l=3:degree+1
        β[l,:] = -x.*β[l-1,:].-(l-2)*β[l-2,:]
    end
    β = β[2:end,:]
    XX = repeat(x,1,N)
    DX = XX-XX'
    DX[I(N)].=1.0
    c = α.*prod(DX,dims=2)
    C = repeat(c,1,N)
    C = C./C'
    Z = 1 ./DX
    Z[I(N)].=0.0
    X = Z'
    X = reshape(X[.~I(N)],N-1,N)
    Y = ones(N-1,N)
    D = I(N)
    for d=1:degree
        Y = cumsum([β[d,:]'; d*Y[1:N-1,:].*X],dims=1)
        D = d*Z.*(C.*repeat(diag(D),1,N)-D)
        D[I(N)]=Y[N,:]
    end
    D = b^degree*D
    return D
end

function her_weights(N::Integer,b::Real=1.0)
    x = her_grid(N,b)
    w = ([diff(x);0]+[0;diff(x)])/2
    return w
end
