using Flux
using Flux: gradient
using Flux.Optimise: update!
using DelimitedFiles, Statistics
using Parameters: @with_kw

# This replicates the housing data example from the Knet.jl readme. Although we
# could have reused more of Flux (see the mnist example), the library's
# abstractions are very lightweight and don't force you into any particular
# strategy.

# Struct to define hyperparameters
@with_kw mutable struct Hyperparams
    lr::Float64 = 0.1		# learning rate
    split_ratio::Float64 = 0.1	# Train Test split ratio, define percentage of data to be used as Test data
end

function get_processed_data(args)
    isfile("housing.data") ||
        download("https://raw.githubusercontent.com/MikeInnes/notebooks/master/housing.data",
            "housing.data")

    rawdata = readdlm("housing.data")'

    # The last feature is our target -- the price of the house.
    split_ratio = args.split_ratio # For the train test split

    x = rawdata[1:13,:]
    y = rawdata[14:14,:]

    # Normalise the data
    x = (x .- mean(x, dims = 2)) ./ std(x, dims = 2)

    # Split into train and test sets
    split_index = floor(Int,size(x,2)*split_ratio)
    x_train = x[:,1:split_index]
    y_train = y[:,1:split_index]
    x_test = x[:,split_index+1:size(x,2)]
    y_test = y[:,split_index+1:size(x,2)]

    train_data = (x_train, y_train)
    test_data = (x_test, y_test)

    return train_data,test_data
end

# Struct to define model
mutable struct model
    W::AbstractArray
    b::AbstractVector
end

# Function to predict output from given parameters
predict(x, m) = m.W*x .+ m.b

# Define the mean squared error function to be used in the loss 
# function. An implementation is also available in the Flux package
# (https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.mse).
meansquarederror(ŷ, y) = sum((ŷ .- y).^2)/size(y, 2)

function train(; kws...)
    # Initialize the Hyperparamters
    args = Hyperparams(; kws...)
    
    # Load the data
    (x_train,y_train),(x_test,y_test) = get_processed_data(args)
    
    # The model
    m = model((randn(1,13)),[0.])
    
    loss(x, y) = meansquarederror(predict(x, m), y) 

    ## Training
    η = args.lr
    θ = params(m.W, m.b)

    for i = 1:500
        g = gradient(() -> loss(x_train, y_train), θ)
        for x in θ
            update!(x, g[x]*η)
        end
        if i%100==0
            @show loss(x_train, y_train)
        end
    end
    
    # Predict the RMSE on the test set
    err = meansquarederror(predict(x_test, m),y_test)
    println(err)
end
##===============mmy
using DifferentialEquations, Plots
## Setup ODE
function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,p)
# Verify ODE solution
sol = solve(prob,Tsit5())
plot(sol)

# Generate data from the ODE
sol = solve(prob,Tsit5(),saveat=0.1)
A = sol[1,:] # length 101 vector
t = 0:0.1:10.0
scatter!(t,A)

using Flux, DiffEqFlux
# Build a neural network that sets the cost as the difference
# from the generated data and 1
p = [2.2, 1.0, 2.0, 0.4]
p = Flux.params(p) # Initial Parameter Vector

function predict_rd() # Our 1-layer neural network
  diffeq_rd(p,prob,Tsit5(),saveat=0.1)[1,:]
end
loss_rd() = sum(abs2,x-1 for x in predict_rd()) # loss function

# Optimize the parameters so the ODE's solution stays near 1
data = Iterators.repeated((), 100)
opt = ADAM(0.1)
Flux.train!(loss_rd, [p], data, opt, cb = cb)

##=====Poisson
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
# 2D PDE
eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)
# Boundary conditions
bcs = [u(0,y) ~ 0.f0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
       u(x,0) ~ 0.f0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0), y ∈ IntervalDomain(0.0,1.0)]
# Discretization
dx = 0.1
# Neural network
dim = 2 # number of dimensions
chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),
                  FastDense(16,1))
discret = PhysicsInformedNN(chain, GridTraining(dx)) # PINNs model
pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
prob = discretize(pde_system,discret)
res = GalacticOptim.solve(prob, Optim.BFGS(); maxiters=1000)
phi = discret.phi

cd(@__DIR__)
train()
