using Distributions


function block_bootstrap_DGP(dfm::DynamicFactorModel, factor_innovations::Array{Float64, 2}, block_size=10)
end
block_bootstrap_DGP(dfm::DynamicFactorModel) = block_bootstrap_DGP(dfm, apply(randn, size(dfm.x)))

function parametric_bootstrap_DGP(dfm::DynamicFactorModel, factor_innovations::Array{Float64, 2})
    # draw x from a parametric bootstrap with the parameters (estimated factors and loadings) given in dfm
    # Note: this is a DGP for the factor equation (x = factors*loadings' + e where e are the innovations) only
    x = dfm.factors*dfm.loadings' + factor_innovations
end
parametric_bootstrap_DGP(dfm::DynamicFactorModel) = parametric_bootstrap_DGP(dfm, apply(randn, size(dfm.x)))  # if no arguments are applied all innovations are assumed standard normal




# static factor models:


function residual_bootstrap(dfm::DynamicFactorModel, B::Int, stat::Function)  # resample residuals
    stats = Array(Float64, B)
    for b in 1:B
        resampled_x = dfm.factors[:, 1:dfm.number_of_factors]*dfm.loadings[:, 1:dfm.number_of_factors]' + dfm.factor_residuals[rand(Distributions.DiscreteUniform(1, length(dfm.y)), size(dfm.x, 1)), :] # TODO: not sure resampling over the T index is correct
        resampled_y = dfm.y  # TODO: resampling y is less interesting. This would be akin to checking a linear model for a break.
        resampled_w = dfm.w
        stats[b] = stat(DynamicFactorModel(resampled_y, resampled_w, resampled_x, dfm.number_of_factors, dfm.number_of_factors_criterion, dfm.factor_type, dfm.targeted_predictors, dfm.number_of_factor_lags, dfm.break_indices))
    end
    stats
end

function wild_bootstrap(dfm::DynamicFactorModel, B::Int, stat::Function)  # resample residuals and multiply by random Variable with mean 0 and variance 1, do that B times and calculate statistic stat
    stats = Array(Float64, B)
    for b in 1:B
        resampled_factor_residuals = dfm.factor_residuals[rand(Distributions.DiscreteUniform(1, length(dfm.y)), size(dfm.x, 1)), :]
        resampled_x = dfm.factors[:, 1:dfm.number_of_factors]*dfm.loadings[:, 1:dfm.number_of_factors]' + apply(vcat, [resampled_factor_residuals[t, :] .* randn() for t in 1:size(resampled_factor_residuals, 1)])
        resampled_y = dfm.y  # TODO: resampling y is less interesting. This would be akin to checking a linear model for a break.
        resampled_w = dfm.w  # TODO: maybe a function for how to resample w?
        stats[b] = stat(DynamicFactorModel(resampled_y, resampled_w, resampled_x, dfm.number_of_factors, dfm.number_of_factors_criterion, dfm.factor_type, dfm.targeted_predictors, dfm.number_of_factor_lags, dfm.break_indices))
    end
    stats
end

#y, x, f, lambdas, epsilon_x = factor_model_DGP(100, 60, 1; model="Breitung_Eickmeier_2011", b=0)
#w = reshape(ones(length(y)), (length(y), 1))
#dfm = DynamicFactorModel(y,w,x,1)
