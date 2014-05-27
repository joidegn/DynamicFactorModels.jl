using Distributions


function sums_squared_residuals(dfm::DynamicFactorModel, break_period, variable_index)  # gets residuals from OLS regression of x on factors for both subsamples for variable_index (i.e. column of the x matrix)
    x1, y1 = dfm.factors[1:break_period, :], dfm.x[1:break_period, variable_index]
    x2, y2 = dfm.factors[break_period+1:end, :], dfm.x[break_period+1:end, variable_index]
    return( sum(((eye(break_period) - x1*inv(x1'x1)x1')y1).^2), sum(((eye(length(y2)) - x2*inv(x2'x2)x2')y2).^2) )
end


# static factor Chow-tests Breitung, Eickmeier (2011)

# LR test
function LR_test(dfm::DynamicFactorModel, break_period::Int64, variable_index)  # TODO: in 2009 version of their paper Breitung and Eickmeier note that we can also use residuals from PCA in sub periods.
    T = size(dfm.x, 1)
    likelihood_ratio = T * (log(sum(dfm.factor_residuals[:, variable_index].^2)) - log(apply(+, (sums_squared_residuals(dfm, break_period, variable_index)))))
    critical_value = quantile(Distributions.Chisq(dfm.number_of_factors), 0.95)  # TODO: we have only estimated the number of factors and estimates are not quite stable
    return likelihood_ratio
end


function Wald_test(dfm::DynamicFactorModel, break_period)
    design_matrix = hcat(dfm.factors, dfm.factors.*[zeros(break_period), ones(length(dfm.y)-break_period)])
    for i in 1:size(dfm.x,2)
        ols_estimates = inv(design_matrix'design_matrix)design_matrix'dfm.x[:, i]
        residuals = dfm.x[:, i] - design_matrix * ols_estimates
        # calcualte variance-covariance matrix according to White(1980)
        coefficient_covariance = inv(design_matrix'design_matrix)*(design_matrix'*diagm(residuals.^2)*design_matrix)*inv(design_matrix'design_matrix)
        # negative covariance estimates?
        # TODO: unfinished
    end
end


function LM_test(dfm::DynamicFactorModel, break_period)
    design_matrix = hcat(dfm.factors, dfm.factors.*[zeros(break_period), ones(size(dfm.x, 1)-break_period)])
    lagrange_multipliers = Array(Float64, size(dfm.x, 2))
    for i in 1:size(dfm.x,2)
        ols_estimates = inv(design_matrix'design_matrix)design_matrix'dfm.factor_residuals[:, i]
        residuals = dfm.factor_residuals[:, i] - design_matrix * ols_estimates
        R_squared = 1 - sum(residuals.^2) / sum(dfm.factor_residuals[:, i].^2)
        lagrange_multipliers[i] = size(x, 1) * R_squared
    end
end
