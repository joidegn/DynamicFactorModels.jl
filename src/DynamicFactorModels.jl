module DynamicFactorModels

#using GLM
using DimensionalityReduction
using Distributions
using GLMNet

# text represenation
import Base.show  # TODO: add show function for DynamicFactormodel
#show()

# allow formulae to be updated by "adding" a string to them  TODO: pull request to DataFrames.jl?
#+(formula::Formula, str::ASCIIString) = Formula(formula.lhs, convert(Symbol, *(string(formula.rhs), " + ", str)))


function generate_ar(params=[0.4, 0.3, 0.2, 0.1], innovations=[], length_series=1004)
    if isempty(innovations)
        innovations = randn(length_series)
    end
    ar = innovations
    for i in (length(params)+1):length_series
        ar_term = (params'*ar[i-length(params):i-1])[1]
        ar[i] = ar[i] + ar_term
    end
    ar
end
function lag_vector{T<:Number}(vec::Array{T,1})
    DataArray([0, vec[1:end-1]], [true, falses(length(vec)-1)])
end
function lag_vector{T<:Number}(vec::DataArray{T,1})
    DataArray([0, vec.data[1:end-1]], [true, vec.na[1:end-1]])
end


function targeted_predictors(y::Array{Float64,1}, w::Matrix{Float64}, x::Matrix{Float64}; thresholding::String="hard")
    # Bai and Ng (2008) -> regress y on w and x and keep only the predictors x which are significant
    # this functions returns their column numbers
    # note that w should maybe consist of a constant (vector of 1s) and lags of y
    #predictors = DataFrame(hcat(y, w, x))  # unfortunately using formulae and the GLM package is a bit clumsy here
    #lm(Formula(convert(Symbol, "x1"), convert(Symbol, join(names(predictors[2:end]), "+"))), predictors)
    #lm(*("x1~", join(names(predictors[2:end]), "+")), predictors)

    design_matrix = hcat(w, x)
    if thresholding == "hard"
        coefficients = inv(design_matrix'*design_matrix)*(design_matrix'*y)  # OLS estimate
        residuals = design_matrix * coefficients
        # calcualte variance-covariance matrix according to White(1980) TODO: argument to choose if robust or not
        coefficient_covariance = inv(design_matrix'design_matrix)*(design_matrix'*diagm(residuals.^2)*design_matrix)*inv(design_matrix'design_matrix)
        t_stat = coefficients./sqrt(diag(coefficient_covariance))
        critical_value = quantile(TDist(length(y)-length(coefficients)), 0.975)
        return abs(t_stat[size(w)[2]+1:end]) .> critical_value  # return significant t-stats associated with x variables only (Bai, Ng, 2008)
    end
    if thresholding == "soft"  # lasso coefficients which are non 0
        #res=lars(design_matrix, y, intercept=true, standardize=true, use_gram=true, maxiter=typemax(Int), lambda_min=0.0, verbose=true)
        res = GLMNet.glmnetcv(design_matrix, y)  # TODO: GLMNet should not be necessary here...
        betas = res.path.betas[:, indmin(res.meanloss)]
        return abs(betas[size(w)[2]+1:end]) .> 0
    end
end

type DynamicFactorModel
    coefficients::Array{Float64, 1}
    coefficient_covariance::Matrix
    design_matrix::Matrix
    t_stats::Array{Float64, 1}
    residuals::Array{Float64, 1}

    function DynamicFactorModel(y::Array{Float64,1}, w::Matrix{Float64}, x::Matrix{Float64}; factor_type::String="principal components", targeted_predictors=1:size(x)[2])
        if factor_type == "principal components"
            pca_res = pca(x[:, targeted_predictors])
            pca_index = pca_res.cumulative_variance .< 0.95  # take 95% of variance TODO: add argument, maybe factor_type is a tuple?
            factors = pca_res.scores[:, pca_index]
        end
        in_model = [trues(size(w)[2]), pca_index]
        design_matrix = hcat(w, factors)
        coefficients = inv(design_matrix'design_matrix)*design_matrix'y
        residuals = y - design_matrix*coefficients
        coefficient_covariance = inv(design_matrix'design_matrix)*(design_matrix'*diagm(residuals.^2)*design_matrix)*inv(design_matrix'design_matrix)
        t_stats = coefficients./sqrt(diag(coefficient_covariance))
        return new(coefficients, coefficient_covariance, design_matrix, t_stats, residuals)
    end
end



end # module
