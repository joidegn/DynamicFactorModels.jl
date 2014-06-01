using Distributions


function sums_squared_residuals(dfm::DynamicFactorModel, break_period, variable_index)  # gets residuals from OLS regression of x on factors for both subsamples for variable_index (i.e. column of the x matrix)
    x1, y1 = dfm.factors[1:break_period, 1:dfm.number_of_factors], dfm.x[1:break_period, variable_index]
    x2, y2 = dfm.factors[break_period+1:end, 1:dfm.number_of_factors], dfm.x[break_period+1:end, variable_index]
    factors1, loadings1, number_of_factors1 = calculate_factors(dfm.x[1:break_period, :])
    factors2, loadings2, number_of_factors2 = calculate_factors(dfm.x[break_period+1:end, :])
    #return( sum((dfm.x[1:break_period, variable_index] - (factors1[:, 1:dfm.number_of_factors]*loadings1[:, dfm.number_of_factors]')[:, variable_index]).^2), sum((dfm.x[break_period+1:end, variable_index] - (factors2[:, 1:dfm.number_of_factors]*loadings2[:, dfm.number_of_factors]')[:, variable_index]).^2) )  # TODO: number of factors is likely to be higher if a structural break is present. --> number of factors should be reestimated for subperiods
    return( sum(((eye(break_period) - x1*inv(x1'x1)x1')y1).^2), sum(((eye(length(y2)) - x2*inv(x2'x2)x2')y2).^2) )   # use OLS for subperiods as in Breitung Eickmeier 2011 or Factor Model as in the 2009 version of the same paper?
end


# static factor Chow-tests Breitung, Eickmeier (2011)

# LR test
function LR_test(dfm::DynamicFactorModel, break_period::Int64, variable_index::Int64)  # TODO: in 2009 version of their paper Breitung and Eickmeier note that we can also use residuals from OLS regressions because they are asymptotically the same as PCA
    T = size(dfm.x, 1)
    likelihood_ratio = T * (log(sum(dfm.factor_residuals[:, variable_index].^2)) - log(apply(+, sums_squared_residuals(dfm, break_period, variable_index))))
    #critical_value = quantile(Distributions.Chisq(dfm.number_of_factors), 0.95)
    return likelihood_ratio
end

function Wald_test(dfm::DynamicFactorModel, break_period::Int64, variable_index::Int64)
    design_matrix = hcat(dfm.factors[:, 1:dfm.number_of_factors], dfm.factors[:, 1:dfm.number_of_factors].*[zeros(break_period), ones(length(dfm.y)-break_period)])
    ols_estimates = inv(design_matrix'design_matrix)design_matrix'dfm.x[:, variable_index]
    residuals = dfm.x[:, variable_index] - design_matrix * ols_estimates
    # calculate variance-covariance matrix according to White(1980) TODO: maybe this should be maximum likelihood estimation of var-cov matrix (i.e. negative inverse of information matrix)
    coefficient_covariance = inv(design_matrix'design_matrix)*(design_matrix'*diagm(residuals.^2)*design_matrix)*inv(design_matrix'design_matrix)
    wald_stat = (ols_estimates[dfm.number_of_factors+1:end]'inv(coefficient_covariance[dfm.number_of_factors+1:end, dfm.number_of_factors+1:end])ols_estimates[dfm.number_of_factors+1:end])[1]
    return wald_stat
end

function LM_test(dfm::DynamicFactorModel, break_period::Int64, variable_index::Int64)
    design_matrix = hcat(dfm.factors[:, 1:dfm.number_of_factors], dfm.factors[:, 1:dfm.number_of_factors].*[zeros(break_period), ones(length(dfm.y)-break_period)])
    ols_estimates = inv(design_matrix'design_matrix)design_matrix'dfm.factor_residuals[:, variable_index]
    residuals = dfm.factor_residuals[:, variable_index] - design_matrix * ols_estimates
    R_squared = 1 - sum(residuals.^2) / sum(dfm.factor_residuals[:, variable_index].^2)
    lagrange_multiplier = size(dfm.x, 1) * R_squared
    return lagrange_multiplier
end
