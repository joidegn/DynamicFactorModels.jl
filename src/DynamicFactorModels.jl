module DynamicFactorModels

#using GLM
using DataArrays
using DataFrames
using Distributions
using Gadfly
#using GLMNet
#using Fred

# text represenation
import StatsBase.predict
#show()

export DynamicFactorModel, predict, lag_vector, targeted_predictors, factor_model_DGP, normalize, pseudo_out_of_sample_forecasts 

include("utils.jl")

function factor_model_DGP(T::Int, N::Int, r::Int; model::String="Bai_Ng_2002")  # T: length of series, N: number of variables, r dimension of factors
    if model=="Breitung_Kretschmer_2004"  # factors follow AR(1) process
        # TODO
    end
    if model=="Breitung_Eickmeier_2011"
        bs = [1, 0.3, 0.5, 0.7, 1]
        # TODO: unfinished, untested
        sigma = rand(Uniform(0.5, 1.5), N)
        f = randn(T, r)  # not specified in the paper
        lambda = randn(r, N) .+ 1  # N(1,1)  TODO: insert a break here? (see page 72 of Breitung)
        epsilon = randn(T, N)*sigma
        x = lambda .* f + epsilon  # TODO this is wrong
        # TODO: inconsistency in naming schemes of Breitung and Bai, Ng. Take a look at Stock, Watson (2002)
    end

    if model=="Bai_Ng_2002"
        f = randn(T, r)
        lambda = randn(N, r)
        theta = r  # base case in Bai, Ng 2002
        epsilon_x = sqrt(theta)*randn(T, N)  # TODO: we could replace errors with AR(p) errors?
        x = f * lambda' + epsilon_x
        beta = rand(Uniform(), r)
        epsilon_y = randn(T)  # TODO: what should epsilon be?
        y = f*beta + epsilon_y # TODO: what should beta be?
        return(y, x, f, lambda, epsilon_x, epsilon_y)
    end
end

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
        residuals = design_matrix * coefficients - y
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


# TODO: approximate principal components rather than principal components could be more efficient for N > T
# TODO: time series structure of factors: how many lags of the factors to include?
type DynamicFactorModel
    coefficients::Array{Float64, 1}
    coefficient_covariance::Array{Float64, 2}
    y::Array{Float64, 1}  # regressor
    w::Array{Float64, 2}  # regressands (e.g. lags of y, constant and variables known to affect y directly)
    x::Array{Float64, 2}  # variables to calculate factors from
    design_matrix::Array{Float64, 2}  # w and the factors
    targeted_predictors::BitArray
    number_of_factors::Int  # columns of factors we use (which capture a certain percentage of the variation e.g.)
    rotation::Array{Float64, 2}  # rotation matrix to calculate factors from x  (inverse of transpose of factor loadings lambda? Yes and also equal to factor loadings due to ortogonality?!)
    t_stats::Array{Float64, 1}
    residuals::Array{Float64, 1}  # residuals of the regression of y on w and the factors
    factor_residuals::Array{Float64, 2}  # residuals from the factor estimation  x = factors * lambda
    factor_type::String
    number_of_factors_criterion::String
    number_of_factors_criterion_value::Float64

    # workhorse method, other methods exist which e.g. determine some of the arguments and then call this function
    function DynamicFactorModel(y::Array{Float64,1}, w::Matrix{Float64}, x::Matrix{Float64}, number_of_factors_criterion::String="", number_of_factors::Int64=minimum(size(x)), factor_type::String="principal components", targeted_predictors::Range1{Int64}=1:size(x, 2), number_of_factor_lags::Int64=0, break_indices::Array{Int64, 1}=Array(Int64, 0))
        # TODO: so far we only have a static factor model. factors need to be defined as in Stock, Watson (2010) page 3
        # TODO: include lagged factors into the regression (how many?)
        factors, loadings, number_of_factors = calculate_factors(x, factor_type, targeted_predictors, number_of_factors)
        # TODO: check if correct factors are used (p 1136 on Bai Ng 2006)
        rotation = inv(loadings')  # rotation Matrix which can be used to rotate X into factor space (follows from X = F*loadings'
        factor_residuals = x .- factors[:, 1:number_of_factors] * loadings[:, 1:number_of_factors]'  # TODO: active factors are considered. Is that correct?

        # TODO: for lags we have to regress residuals on its lags and use info criterio to choose lag length p. this should be offloaded into another method
        # TODO: solve chicken and egg problem between number_of_factors and number_of_factor_lags. Both require residuals. --> maybe select one after the other repeatedly until convergence?
        # TODO: do the above e.g. factors, loadings, number_of_factors = augment_factors_by_lags(factors, number_of_factor_lags)  # TODO: do we have to 
        # for structural breaks we augment the design matrix with factors multiplied with index which is 0 from t in [1, ..., t*-1] 1 from T in [t*, ..., T]

        design_matrix = make_factor_model_design_matrix(w, factors, number_of_factors, number_of_factor_lags)
        coefficients = inv(design_matrix'design_matrix)*design_matrix'y
        residuals = y - design_matrix*coefficients
        #hat_matrix = design_matrix*inv(design_matrix'design_matrix)*design_matrix'
        #residual_variance = (residuals.^2)./(1.-diag(hat_matrix))  # HC 2
        residual_variance = residuals.^2  # HC 0
        coefficient_covariance = inv(design_matrix'design_matrix)*(design_matrix'diagm(residual_variance)design_matrix)*inv(design_matrix'design_matrix)
        #coefficient_covariance = inv(design_matrix'design_matrix)*(1/length(residual_variance)*sum(residual_variance))
        if any(diag(coefficient_covariance).<0)
            warn("negative variance estimates. This is so haywood")
            if iscomplex(design_matrix)
                warn("design_matrix has complex entries")
            end
            t_stats = zeros(size(design_matrix, 2))
        else
            t_stats = coefficients./(sqrt(diag(coefficient_covariance)))
        end
        #t_stats = try
        #    coefficients./(sqrt(diag(coefficient_covariance)))
        #catch error
        #    warn("haywood case or whatever the fuck is going on. T-stats set to 0")
        #    zeros(length(coefficients))
        #end

        if number_of_factors_criterion == ""
            return(new(coefficients, coefficient_covariance, y, w, x, design_matrix, targeted_predictors, number_of_factors, rotation, t_stats, residuals, factor_residuals, factor_type, number_of_factors_criterion))
        else
            return(calculate_criterion(new(coefficients, coefficient_covariance, y, w, x, design_matrix, targeted_predictors, number_of_factors, rotation, t_stats, residuals, factor_residuals, factor_type, number_of_factors_criterion)))
        end
    end

    function DynamicFactorModel(y::Array{Float64,1}, w::Matrix{Float64}, x::Matrix{Float64}, number_of_factors_criterion::String, factor_type::String="principal components", targeted_predictors::Range1=1:size(x, 2), number_of_lags::Int=0, break_indices::Array{Int, 1}=Array(Int, 0))
        # number of factors to include is not given but a criterion --> we use the criterion to determine the number of factors
        # if number of factors are to be calculated according to a criterion we need to estimate the model until criterion is optimal
        # we can probably stop when the criterion starts growing, I think the criterion functions are convex
        # TODO: what to do with chicken and egg problem: next line tries to calculate number of factors but 
        models = [apply(DynamicFactorModel, (y, w, x, number_of_factors_criterion, number_of_factors, factor_type, targeted_predictors, number_of_lags, break_indices)) for number_of_factors in 1:size(x, 2)]  # we keep all the models in memory which can be a problem depending on the dimensions of x. TODO: will refactor later when debugging and testing is done
        criteria = [model.number_of_factors_criterion_value for model in models]
        return models[indmin(criteria)]  # keep the model with the best information criterium
    end
end

# calculate the factors and the rotation matrix to transform data into the space spanned by the factors
function calculate_factors(x::Matrix, factor_type::String="principal components", targeted_predictors=1:size(x, 2), number_of_factors=minimum(size(x)))
    T, N = size(x)  # T: time dimension, N cross-sectional dimension
    if factor_type == "principal components"
        #pca_res = pca(x[:, targeted_predictors]; center=false, scale=false)
        # see Stock, Watson (1998)
        if T >= N
            eigen_values, eigen_vectors = eig(x'x)
            eigen_values, eigen_vectors = reverse(eigen_values), eigen_vectors[:, size(eigen_vectors, 2):-1:1]  # reverse the order from highest to lowest eigenvalue
            loadings = sqrt(N) * eigen_vectors'  # we may rescale the eigenvectors say Bai, Ng 2002
            factors = x*inv(loadings')/N  # this is from Bai, Ng 2002
            #factors = factors*(factors'factors/T)^(1/2)  # TODO: not sure it is correct to rescale like this (see Bai, Ng 2002 p.198)
            #factors = 1/N*x*loadings'  # this is from Stock, Watson 2010
        end
        if N > T
            eigen_values, eigen_vectors = eig(x*x'/(T*N))
            eigen_values, eigen_vectors = reverse(eigen_values), eigen_vectors[:, size(eigen_vectors, 2):-1:1]  # reverse the order from highest to lowest eigenvalue
            factors = sqrt(T) * eigen_vectors  # dimension of factors is Txr where r can be 1, ..., T
            loadings = x'factors/T  # see e.g. Bai 2003 p.6 or Breitung, Eickmeier 2011 p. 80. Dimension of loadings is Nxr where r here is also N
            # resultingly factors*loadings' estimates x
        end
        max_factor_number = minimum(size(x))
    elseif factor_type == "squared principal components"  # include squares of X
        pca_res = pca([x[:, targeted_predictors] x[:, targeted_predictors].^2]; center=false, scale=false)
        max_factor_number = minimum([size(x, 1), size(x,2)*2])
    elseif factor_type == "quadratic principal components"  # include squares of X and interaction terms - better only use in combination with targeted_predictors
        pca_cols = x[:, targeted_predictors]  # columns to use for principal components
        for i in 1:size(x[:, targeted_predictors], 2)
            for j in 1:size(x[:, targeted_predictors], 2)
                pca_cols = hcat(pca_cols, x[:, i].*x[:, j])
            end
        end
        max_factor_number = minimum(size(pca_cols))
        pca_res = pca(pca_cols; center=false, scale=false)
        # TODO: or replace with Bai Ng 2002: Determining the number of factors in approximate factor models
    end
    if number_of_factors > max_factor_number
        number_of_factors = max_factor_number
        warn("can not estimate more than `minimum(size(x))` factors with $factor_type. Number of factors set to $number_of_factors")
    end
    return factors, loadings, number_of_factors  # TODO: not sure about the sqrt(...) I think this is only for T>N (see Bai Ng 2002 p 198)
end


function calculate_criterion(dfm::DynamicFactorModel)
    number_of_factors_criterion = dfm.number_of_factors_criterion
    dfm.number_of_factors_criterion_value = eval(symbol("criterion_$number_of_factors_criterion"))(dfm)
    return(dfm)
end
factor_residual_variance(dfm::DynamicFactorModel) = sum(dfm.factor_residuals.^2)/apply(*, size(x))  # see page 201 of Bai Ng 2002
#factor_residual_variance(dfm::DynamicFactorModel) = sum(mapslices(x->x'x/length(x), dfm.factor_residuals, 1))/size(dfm.x, 2)  # the same as above
# and var(dfm.factor_residuals) is approximately the same as well



function Base.show(io::IO, dfm::DynamicFactorModel)
    @printf io "Dynamic Factor Model\n"
    @printf io "Dimensions of X: %s\n" size(dfm.x)
    @printf io "Number of factors used: %s\n" dfm.number_of_factors
    @printf io "Factors calculated by: %s\n" dfm.factor_type
end

# prediction needs w and (original i.e. non-transformed) x
function predict(dfm::DynamicFactorModel, w, x)
    design_matrix = hcat(w, get_factors(dfm, x))
    return design_matrix*dfm.coefficients
end


# transforms x to the space spanned by the factors and optionally only selects active factors
#   type="active" returns only the active factors (which explain enough of the variance)
function get_factors(dfm::DynamicFactorModel, x::Matrix, factors="active")  # TODO: do I have to divide by N if T>N? (see Bai, Ng 2002 p. 198)
    (normalize(x[:, dfm.targeted_predictors], (mean(dfm.x), std(dfm.x)))*dfm.rotation)[:, factors=="active" ? (1:dfm.number_of_factors) : (1:end)]
end

            
data = readtable("/home/joi/Documents/Konstanz/Masterarbeit/data/1959-2014_normalized.csv")
data_matrix = reduce(hcat, [convert(Array{Float64}, col) for col in data.columns[2:size(data.columns, 1)]])
#ids = map(string, names(data)[2:end])
#titles = series_titles(ids)  # TODO: does not work at the moment because ICU.jl and with it Requests.jl seems to be broken
T = size(data_matrix, 1) - 4  # we include 4 lags
y = data_matrix[:,1]  # TODO: this is not something we actually want to predict...
data_matrix = data_matrix[:, 2:end]
lag1 = lag_vector(y)
lag2 = lag_vector(lag1)
lag3 = lag_vector(lag2)
lag4 = lag_vector(lag3)
y = y[5:end]
w = hcat(ones(T), array(lag1[5:end]), array(lag2[5:end]), array(lag3[5:end]), array(lag4[5:end]))
x = data_matrix[5:end, :]

## need some problematic data for debugging something:
#for problem_index in 1:200
#    println("index: ", problem_index)
#    problem_x = x[problem_index:end, :]
#    problem_y = y[problem_index:end]
#    problem_w = w[problem_index:end, :]
#    model = DynamicFactorModel(problem_y, problem_w, problem_x)
#end

#@time predictions = pseudo_out_of_sample_forecasts(DynamicFactorModel, y, w, x)
#mse = mean((predictions.-y[end-200, end]).^2)
#predictions_frame = DataFrame(
#    predictions=vcat(y[end-199:end], predictions, predictions_targeted, predictions_ols, predictions_average),
#    method=split(^("true value,", 200) * ^("Static Factor Model,", 200) * ^("targeted Static Factor Model,", 200) * ^("OLS,", 200) * ^("Average,", 200), ",", 0, false)
#)
#set_default_plot_format(:png)
#display(predictions_plot)

include("criteria.jl")  # defines the criteria in Bai and Ng 2002
include("chowtest.jl")

end # module
