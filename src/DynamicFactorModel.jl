using DimensionalityReduction


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
    number_of_factors::Int64  # columns of factors we use (which capture a certain percentage of the variation e.g.)
    number_of_factor_lags::Int64  # columns of factors we use (which capture a certain percentage of the variation e.g.)
    break_indices::Array{Int, 1}  # gives indices of breakpoints which are to be taken into account in estimation
    #rotation::Array{Float64, 2}  # rotation matrix to calculate factors from x (inverse of transpose of factor loadings lambda? Yes and also equal to factor loadings due to ortogonality?!)
    factors::Array{Float64, 2}
    loadings::Array{Float64, 2}
    t_stats::Array{Float64, 1}
    residuals::Array{Float64, 1}  # residuals of the regression of y on w and the factors
    factor_residuals::Array{Float64, 2}  # residuals from the factor estimation  x = factors * lambda
    factor_type::String
    number_of_factors_criterion::String
    number_of_factors_criterion_value::Float64

    # workhorse method, other methods exist which e.g. determine some of the arguments and then call this function
    function DynamicFactorModel(y::Array{Float64,1}, w::Matrix{Float64}, x::Matrix{Float64}, number_of_factors::Int64=ceil(minimum(size(x))/2), number_of_factors_criterion::String="", factor_type::String="principal components", targeted_predictors::BitArray=trues(size(x, 2)), number_of_factor_lags::Int64=0, break_indices::Array{Int64, 1}=Array(Int64, 0))
        # TODO: so far we only have a static factor model. factors need to be defined as in Stock, Watson (2010) page 3
        # TODO: include lagged factors into the regression (how many?)
        factors, loadings, number_of_factors = calculate_factors(x, factor_type, targeted_predictors, number_of_factors)
        # TODO: check if correct factors are used (p 1136 on Bai Ng 2006)
        #rotation = inv(loadings')  # rotation Matrix which can be used to rotate X into factor space (follows from X = F*loadings'
        factor_residuals = x .- factors[:, 1:number_of_factors] * loadings[:, 1:number_of_factors]'  # TODO: active factors are considered. Is that correct?
        #factor_residuals = x - x*loadings[:, 1:number_of_factors]*inv(loadings[:, 1:number_of_factors]'loadings[:, 1:number_of_factors])*loadings[:, 1:number_of_factors]' # rescaled version from french dude
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
            t_stats = zeros(size(design_matrix, 2))
        else
            t_stats = coefficients./(sqrt(diag(coefficient_covariance)))
        end

        return(calculate_criterion(new(coefficients, coefficient_covariance, y, w, x, design_matrix, targeted_predictors, number_of_factors, number_of_factor_lags, break_indices, factors, loadings, t_stats, residuals, factor_residuals, factor_type, number_of_factors_criterion)))
    end

    function DynamicFactorModel(y::Array{Float64,1}, w::Matrix{Float64}, x::Matrix{Float64}, number_of_factors_criterion::String, factor_type::String="principal components", targeted_predictors::BitArray=trues(size(x, 2)), number_of_lags::Int=0, break_indices::Array{Int, 1}=Array(Int, 0))
        max_factors = int(ceil(minimum(size(x))/2))
        println("finding optimal DFM (with maximum number of factors=", max_factors, ")")
        # number of factors to include is not given but a criterion --> we use the criterion to determine the number of factors
        # if number of factors are to be calculated according to a criterion we need to estimate the model until criterion is optimal
        # we can probably stop when the criterion starts growing, I think the criterion functions are convex
        # TODO: what to do with chicken and egg problem: next line tries to
        # calculate number of factors using among other things the residuals
        # but number_of_factor lags depends on residuals as well
        models = [apply(DynamicFactorModel, (y, w, x, number_of_factors, number_of_factors_criterion, factor_type, targeted_predictors, number_of_lags, break_indices)) for number_of_factors in 1:max_factors]  # we keep all the models in memory which can be a problem depending on the dimensions of x. TODO: will refactor later when debugging and testing is done
        criteria = [model.number_of_factors_criterion_value for model in models]
        println("criteria_values: ", criteria)
        return models[indmin(criteria[1:max_factors])]  # keep the model with the best information criterion
    end
end


# calculate the factors and the rotation matrix to transform data into the space spanned by the factors
function calculate_factors(x::Matrix, factor_type::String="principal components", targeted_predictors=1:size(x, 2), number_of_factors=ceil(minimum(size(x))/2))
    T, N = size(x)  # T: time dimension, N cross-sectional dimension
    if factor_type == "principal components"
        #pca_res = pca(x[:, targeted_predictors]; center=false, scale=false)
        # see Stock, Watson (1998)
        if T >= N
            #pca_res = pca(x; center=false, scale=false)
            #loadings = pca_res.rotation  # actually inverse of transpose but this is the same due to ortogonality
            #factors = pca_res.scores

            eigen_values, eigen_vectors = eig(x'x)
            eigen_values, eigen_vectors = reverse(eigen_values), eigen_vectors[:, size(eigen_vectors, 2):-1:1]  # reverse the order from highest to lowest eigenvalue
            loadings = sqrt(N) * eigen_vectors  # we may rescale the eigenvectors say Bai, Ng 2002
            factors = x*loadings/N  # this is from Bai, Ng 2002 except that for me the inverse of the loadings matrix primed is not the same as the loadings matrix
            #factors = factors*(factors'factors/T)^(1/2)  # TODO: not sure it is correct to rescale like this (see Bai, Ng 2002 p.198)
            #factors = 1/N*x*loadings'  # this is from Stock, Watson 2010
            #loadings = (x'x)*loadings/(N*T)  # and this is from C. Hurlin from University of Orléans... rescales factor loadings to have the same variance as x
        end
        if N > T
            eigen_values, eigen_vectors = eig(x*x')
            eigen_values, eigen_vectors = reverse(eigen_values), eigen_vectors[:, size(eigen_vectors, 2):-1:1]  # reverse the order from highest to lowest eigenvalue
            factors = sqrt(T) * eigen_vectors  # sqrt(T) comes from normalization F'F/T = eye(r) where r is number of factors (see Bai 2003), factors are Txr
            loadings = x'factors/T  # see e.g. Bai 2003 p.6 or Breitung, Eickmeier 2011 p. 80. Dimension of loadings is Nxr where r here is also N
            # betahat = loadings * chol(loadings'loadings/N)  # C. Hurlin from University of Orléans rescales like this
        end
        # resultingly factors*loadings' estimates x
        max_factor_number = ceil(minimum(size(x))/2)
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

# transforms x to the space spanned by the factors and optionally only selects active factors
#   type="active" returns only the active factors (which explain enough of the variance)
function get_factors(dfm::DynamicFactorModel, x::Matrix, factors="active")  # TODO: do I have to divide by N if T>N? (see Bai, Ng 2002 p. 198)
    rotation = dfm.loadings * inv(dfm.loadings'dfm.loadings) # simplifies to dfm.loadings if T>N and loadings = inv(loadings')  which makes sense given x = F * loadings'
    (normalize(x[:, dfm.targeted_predictors], (mean(dfm.x), std(dfm.x)))*dfm.rotation)[:, factors=="active" ? (1:dfm.number_of_factors) : (1:end)]
end

function make_factor_model_design_matrix(w, factors, number_of_factors, number_of_factor_lags)
    return(hcat(w, factors[:, 1:number_of_factors]))  # TODO: unfinished business
    design_matrix = hcat(w, factors)
    for lag_num in 1:number_of_factor_lags
        lag_matrix = 1
    end
end

function calculate_criterion(dfm::DynamicFactorModel)
    if dfm.number_of_factors_criterion != ""
        number_of_factors_criterion = dfm.number_of_factors_criterion
        dfm.number_of_factors_criterion_value = eval(symbol("criterion_$number_of_factors_criterion"))(dfm)
    end
    return(dfm)
end

function Base.show(io::IO, dfm::DynamicFactorModel)
    @printf io "Dynamic Factor Model\n"
    @printf io "Dimensions of X: %s\n" size(dfm.x)
    @printf io "Number of factors used: %s\n" dfm.number_of_factors
    @printf io "Factors calculated by: %s\n" dfm.factor_type
    @printf io "Factor model residual variance: %s\n" sum(dfm.factor_residuals.^2)/apply(*, size(dfm.x))
end

# prediction needs w and (normalized) x
function predict(dfm::DynamicFactorModel, w, x)
    design_matrix = hcat(w, get_factors(dfm, x))
    return design_matrix*dfm.coefficients
end

include("criteria.jl")  # defines the criteria in Bai and Ng 2002
