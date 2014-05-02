module DynamicFactorModels

#using GLM
using DataArrays
using DataFrames
using DimensionalityReduction
using Distributions
using GLMNet
#using Fred

# text represenation
import StatsBase.predict
#show()

export DynamicFactorModel, predict, lag_vector, targeted_predictors

# allow formulae to be updated by "adding" a string to them  TODO: pull request to DataFrames.jl?
#+(formula::Formula, str::ASCIIString) = Formula(formula.lhs, convert(Symbol, *(string(formula.rhs), " + ", str)))

function generate_ar(params=[0.4, 0.3, 0.2, 0.1], innovations=[], length_series=1004)  # for testing TODO: remove
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
function lag_matrix{T<:Number}(matr::Array{T, 2})
    DataFrame([lag_vector(matr[:, col]) for col in 1:size(matr, 2)])
end
function lag_matrix(matr::DataFrame)
    DataFrame([lag_vector(matr[:, col]) for col in 1:size(matr, 2)])
end


function factor_model_DGP(T::Int, N::Int, r::Int)  # T: length of series, N: number of variables, r dimension of factors
    # see e.g. Breitung and Eickmeier, 2011 p. 72
    #sigma = rand(Uniform(0.5, 1.5), N)
    #f = randn(T, r)  # not specified in the paper
    #lambda = randn(r, N) .+ 1  # N(1,1)  TODO: insert a break here? (see page 72 of Breitung)
    #epsilon = randn(T, N)*sigma
    #x = lambda .* f + epsilon  # TODO: inconsistency in naming schemes of Breitung and Bai, Ng. Take a look at Stock, Watson (2002)


    # or see Bai, Ng 2002, p. 202
    f = randn(T, r)
    lambda = randn(r, N)
    theta = r  # base case in Bai, Ng 2002
    epsilon_x = sqrt(theta)*randn(T, N)  # TODO: we could replace errors with AR(p) errors?
    x = f * lambda + epsilon_x
    beta = rand(Uniform(), r)
    epsilon_y = randn(T)  # TODO: what should epsilon be?
    y = f*beta + randn(T) # TODO: what should beta be?
    return(y, x, f, lambda, epsilon_x, epsilon_y)
end

normalize(A::Matrix) = (A.-mean(A,1))./std(A,1) # normalize (i.e. center and rescale) Matrix A
normalize(A::Matrix, by) = (A.-by[1])./by[2] # normalize (i.e. center and rescale) Matrix A by given (mean, stddev)-tuple

function targeted_predictors(y::Array{Float64,1}, w::Matrix{Float64}, x::Matrix{Float64}; thresholding::String="hard")
    # Bai and Ng (2008) -> regress y on w and x and keep only the predictors x which are significant
    # this functions returns their column numbers
    # note that w should maybe consist of a constant (vector of 1s) and lags of y
    #predictors = DataFrame(hcat(y, w, x))  # unfortunately using formulae and the GLM package is a bit clumsy here
    #lm(Formula(convert(Symbol, "x1"), convert(Symbol, join(names(predictors[2:end]), "+"))), predictors)
    #lm(*("x1~", join(names(predictors[2:end]), "+")), predictors)

    design_matrix = hcat(w, x)
    if thresholding == "hard"  # TODO: hard thresholding does not work properly, only admits one variable plus the white covariance has negative diagonal elements
        coefficients = inv(design_matrix'*design_matrix)*(design_matrix'*y)  # OLS estimate
        residuals = design_matrix * coefficients
        # calcualte variance-covariance matrix according to White(1980) TODO: argument to choose if robust or not
        coefficient_covariance = inv(design_matrix'design_matrix)*(design_matrix'*diagm(residuals.^2)*design_matrix)*inv(design_matrix'design_matrix)
        num_negative = sum(diag(coefficient_covariance) .< 0)
        if num_negative > 0
            for i in 1:size(coefficient_covariance, 1)
                if coefficient_covariance[i, i] < 0
                   coefficient_covariance[i, i] = -coefficient_covariance[i, i]
               end
            end
            t_stat = coefficients./sqrt(diag(coefficient_covariance))
            println("there were $num_negative negative coefficient variances which have been set positive. THIS IS UNACCEPTABLE!")
        else
            t_stat = coefficients./sqrt(diag(coefficient_covariance))
        end
        critical_value = quantile(TDist(length(y)-length(coefficients)), 0.975)
        println(sum(abs(t_stat[size(w, 2)+1:end]) .> critical_value))
        return abs(t_stat[size(w, 2)+1:end]) .> critical_value  # return significant t-stats associated with x variables only (Bai, Ng, 2008)
    end
    if thresholding == "soft"  # lasso coefficients which are non 0 are used
        #res=lars(design_matrix, y, intercept=true, standardize=true, use_gram=true, maxiter=typemax(Int), lambda_min=0.0, verbose=true)
        res = GLMNet.glmnetcv(design_matrix, y)  # TODO: GLMNet should not be necessary here...
        betas = res.path.betas[:, indmin(res.meanloss)]
        return abs(betas[size(w, 2)+1:end]) .> 0
    end
end


function number_of_factors_cumulative_variance(pca_result, threshold=0.95)
    pca_result.cumulative_variance .< threshold  # take threshold% of variance 
end


type DynamicFactorModel
    coefficients::Array{Float64, 1}
    coefficient_covariance::Matrix
    y::Array{Float64, 1}  # regressor
    w::Matrix  # regressands (e.g. lags of y, constant and variables known to affect y directly)
    x::Matrix  # variables to calculate factors from
    design_matrix::Matrix
    targeted_predictors::BitArray
    factor_columns::BitArray{1}  # columns of factors we use (which capture a certain percentage of the variation)
    rotation::Matrix  # rotation matrix to calculate factors from x
    t_stats::Array{Float64, 1}
    residuals::Array{Float64, 1}
    factor_type::String

    #TODO: number_of_factors: one of "min_MSE", "BIC", ...?, chowtest?  NOTE: BIC and AIK can only be used when T>>N
    function DynamicFactorModel(y::Array{Float64,1}, w::Matrix{Float64}, x::Matrix{Float64}; factor_type::String="principal components", targeted_predictors=1:size(x, 2))
        if factor_type == "principal components"
            pca_res = pca(x[:, targeted_predictors]; center=false, scale=false)
            pca_index = pca_res.cumulative_variance .< 0.95  # take 95% of variance TODO: add argument, maybe factor_type is a tuple?
            # TODO: or replace with Bai Ng 2002: Determining the number of factors in approximate factor models
        elseif factor_type == "squared principal components"  # include squares of X
            pca_res = pca([x[:, targeted_predictors] x[:, targeted_predictors].^2]; center=false, scale=false)
            pca_index = pca_res.cumulative_variance .< 0.95  # take 95% of variance TODO: add argument, maybe factor_type is a tuple?
            # TODO: or replace with Bai Ng 2002: Determining the number of factors in approximate factor models
        elseif factor_type == "quadratic principal components"  # include squares of X and interaction terms - better only use in combination with targeted_predictors
            pca_cols = x[:, targeted_predictors]  # columns to use for principal components
            for i in 1:size(x[:, targeted_predictors], 2)
                for j in 1:size(x[:, targeted_predictors], 2)
                    pca_cols = hcat(pca_cols, x[:, i].*x[:, j])
                end
            end
            pca_res = pca(pca_cols; center=false, scale=false)
            pca_index = pca_res.cumulative_variance .< 0.95  # take 95% of variance
            # TODO: or replace with Bai Ng 2002: Determining the number of factors in approximate factor models
        end
        factors = pca_res.scores  # TODO: check if correct factors are used (p 1136 on Bai Ng 2006)
        factor_columns = pca_index  # the active factors for estimation

        design_matrix = hcat(w, factors[:, factor_columns])  # TODO: add lags of factors (Dynamic Factor Models!)
        coefficients = inv(design_matrix'design_matrix)*design_matrix'y
        residuals = y - design_matrix*coefficients
        coefficient_covariance = inv(design_matrix'design_matrix)*(design_matrix'*diagm(residuals.^2)*design_matrix)*inv(design_matrix'design_matrix)
        t_stats = coefficients./sqrt(diag(coefficient_covariance))
        return new(coefficients, coefficient_covariance, y, w, x, design_matrix, targeted_predictors, factor_columns, pca_res.rotation, t_stats, residuals, factor_type)
    end

    function DynamicFactorModel(y::Array{Float64,1}, w::Matrix{Float64}, x::Matrix{Float64}, number_of_factors_criterion::String; factor_type::String="principal components", targeted_predictors=1:size(x, 2))
        # if number of factors are to be calculated according to a criterion we need to estimate the model until criterion is optimal
        for number_of_factors in 1:size(x, 2)  # TODO: this will take forever
            res = DynamicFactorModel(y, w, x; factor_type=factor_type, targeted_predictors=targeted_predictors)
            criterion = calculate_criterion(res, number_of_factors_criterion)
        end
    end
end

norm_vector{T<:Number}(vec::Array{T, 1}) = vec./norm(vec) # makes vector unit norm
norm_matrix{T<:Number}(mat::Array{T, 2}) = mapslices(norm_vector, mat, 2)  # call norm_vector for each column

calculate_criterion(res, number_of_factors_criterion) = res |> eval(symbol("criterion_$number_of_factors_ctierion"))
residual_variance(dfm::DynamicFactorModel) = 1/size(dfm.x, 2) * sum(dfm.residuals'dfm.residuals./T)
#criterion_PCp1(res) = res.


function Base.show(io::IO, dfm::DynamicFactorModel)
    @printf io "Dynamic Factor Model\n"
    @printf io "Dimensions of X: %s\n" size(dfm.x)
    @printf io "Number of factors used: %s\n" sum(dfm.factor_columns)
    @printf io "Factors calculated by: %s\n" dfm.factor_type
end

# prediction needs w and (original i.e. non-transformed) x
function predict(dfm::DynamicFactorModel, w, x)
    design_matrix = hcat(w, get_factors(dfm, x))
    return design_matrix*dfm.coefficients
end


# transforms x to the space spanned by the factors and optionally only selects active factors
#   type="active" returns only the active factors (which explain enough of the variance)
function get_factors(dfm::DynamicFactorModel, x::Matrix, factors="active")
    (normalize(x[:, dfm.targeted_predictors], (mean(dfm.x), std(dfm.x)))*dfm.rotation)[:, factors=="active" ? dfm.factor_columns : (1:end)]
end


function make_forecasts(y, w, x; num_predictions=200)
    # one step ahead pseudo out-of-sample forecasts

    predictions = zeros(num_predictions)
    predictions_targeted = zeros(num_predictions)
    predictions_ols = zeros(num_predictions)
    for date_index in T-num_predictions+1:T
        println("date index: $date_index")
        let y=y[1:date_index], x=x[1:date_index, :], w=w[1:date_index, :]  # y, x and w are updated so its easier for humans to read the next lines
            let newx=x[end, :], neww=w[end, :], newy=y[end], y=y[1:end-1], x=x[1:end-1, :], w=w[1:end-1, :]  # pseudo-one step ahead (keeps notation clean in the following lines)
            #y=y[1:date_index]; x=x[1:date_index, :]; w=w[1:date_index, :]
            #newx=x[end, :]; neww=w[end, :]; newy=y[end]; y=y[1:end-1]; x=x[1:end-1, :]; w=w[1:end-1, :]
                res = DynamicFactorModel(y, w, x)
                res_targeted = DynamicFactorModel(y, w, x; targeted_predictors=targeted_predictors(y, w, x; thresholding="soft"))
                beta_ols = inv(hcat(w,x)'hcat(w,x))*hcat(w,x)'y
                predictions[date_index-(T-num_predictions)] = (newy-predict(res, neww, newx))[1]
                predictions_targeted[date_index-(T-num_predictions)] = (newy-predict(res_targeted, neww, newx))[1]
                predictions_ols[date_index-(T-num_predictions)] = newy-(hcat(neww,newx)*beta_ols)[1]
            end
        end
    end
    return(predictions, predictions_targeted, predictions_ols)
end

#data = readtable("/home/joi/Documents/Konstanz/Masterarbeit/data/1959-2014_normalized.csv")
#data_matrix = reduce(hcat, [convert(Array{Float64}, col) for col in data.columns[2:size(data.columns, 1)]])
##ids = map(string, names(data)[2:end])
##titles = series_titles(ids)  # TODO: does not work at the moment because ICU.jl and with it Requests.jl seems to be broken
#T = size(data_matrix, 1) - 4  # we include 4 lags
#y = data_matrix[:,1]  # TODO: this is not something we actually want to predict...
#data_matrix = data_matrix[:, 2:end]
#lag1 = lag_vector(y)
#lag2 = lag_vector(lag1)
#lag3 = lag_vector(lag2)
#lag4 = lag_vector(lag3)
#y = y[5:end]
#w = hcat(ones(T), array(lag1[5:end]), array(lag2[5:end]), array(lag3[5:end]), array(lag4[5:end]))
#x = data_matrix[5:end, :]
#
#predictions, predictions_targeted, predictions_ols = make_forecasts(y, w, x)
#
#mse = mean((predictions.-y[end-200, end]).^2)
#mse_targeted = mean((predictions_targeted.-y[end-200, end]).^2)
#mse_ols = mean((predictions_ols.-y[end-200, end]).^2)



include("chowtest.jl")  # less essential code lives abroad

end # module
