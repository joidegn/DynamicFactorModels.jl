module DynamicFactorModels

#using GLM
using DataArrays
using DataFrames
using Distributions
#using Gadfly
# using DimensionalityReduction
#using GLMNet
#using Fred

# text represenation
import StatsBase.predict
#show()

export DynamicFactorModel, predict, calculate_factors,
    lag_vector, factor_model_DGP, normalize, pseudo_out_of_sample_forecasts,
    targeted_predictors,
    wild_bootstrap, residual_bootstrap,
    LR_test, LM_test, Wald_test

include("utils.jl")
include("DynamicFactorModel.jl")
include("targeted_predictors.jl")
include("bootstrap.jl")
include("chowtest.jl")

end # module
