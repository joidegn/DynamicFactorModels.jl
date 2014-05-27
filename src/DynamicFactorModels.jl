module DynamicFactorModels

#using GLM
using DataArrays
using DataFrames
using Distributions

# text represenation
import StatsBase.predict
#show()

export DynamicFactorModel, predict,
    lag_vector, factor_model_DGP, normalize, pseudo_out_of_sample_forecasts,
    targeted_predictors,
    wild_bootstrap, residual_bootstrap

include("utils.jl")
include("DynamicFactorModel.jl")
include("targeted_predictors.jl")
include("bootstrap.jl")
include("chowtest.jl")

end # module
