



factor_residual_variance(dfm::DynamicFactorModel) = sum(dfm.factor_residuals.^2)/apply(*, size(x))  # see page 201 of Bai Ng 2002
#factor_residual_variance(dfm::DynamicFactorModel) = sum(mapslices(x->x'x/length(x), dfm.factor_residuals, 1))/size(dfm.x, 2)  # the same as above
# and var(dfm.factor_residuals) is approximately the same as well




function criterion_cumulative_variance(pca_result, threshold=0.95)   # simply use factors until a certain threshold of variance is reached
    pca_result.cumulative_variance .< threshold  # take threshold% of variance 
end

# PCp criteria as defined on page 201 of Bai and Ng 2002
function criterion_PCp1(dfm::DynamicFactorModel)
    dfm_unrestricted = DynamicFactorModel(dfm.y, dfm.w, dfm.x)
    N_plus_T_by_NT = apply(+, size(dfm.x))/apply(*, size(dfm.x))
    factor_residual_variance(dfm) + dfm.number_of_factors*factor_residual_variance(dfm_unrestricted)*(N_plus_T_by_NT)*log(N_plus_T_by_NT^-1)
end
function criterion_PCp2(dfm::DynamicFactorModel)
    dfm_unrestricted = DynamicFactorModel(dfm.y, dfm.w, dfm.x)
    N_plus_T_by_NT = apply(+, size(dfm.x))/apply(*, size(dfm.x))
    factor_residual_variance(dfm) + sum(dfm.number_of_factors)*factor_residual_variance(dfm_unrestricted)*(N_plus_T_by_NT)*log(minimum(size(dfm.x)))
end
function criterion_PCp3(dfm::DynamicFactorModel)
    dfm_unrestricted = DynamicFactorModel(dfm.y, dfm.w, dfm.x)
    N_plus_T_by_NT = apply(+, size(dfm.x))/apply(*, size(dfm.x))
    factor_residual_variance(dfm) + sum(dfm.number_of_factors)*factor_residual_variance(dfm_unrestricted)*log(minimum(size(dfm.x)))/minimum(size(dfm.x))
end


# ICp criteria as defined on page 201 of Bai and Ng 2002
function criterion_ICp1(dfm::DynamicFactorModel)
    N_plus_T_by_NT = apply(+, size(dfm.x))/apply(*, size(dfm.x))
    log(factor_residual_variance(dfm)) + sum(dfm.number_of_factors)*(N_plus_T_by_NT)*log(N_plus_T_by_NT^-1)
end
function criterion_ICp2(dfm::DynamicFactorModel)
    N_plus_T_by_NT = apply(+, size(dfm.x))/apply(*, size(dfm.x))
    log(factor_residual_variance(dfm)) + sum(dfm.number_of_factors)*(N_plus_T_by_NT)*log(minimum(size(dfm.x)))
end
function criterion_ICp3(dfm::DynamicFactorModel)
    N_plus_T_by_NT = apply(+, size(dfm.x))/apply(*, size(dfm.x))
    log(factor_residual_variance(dfm)) + sum(dfm.number_of_factors)*log(minimum(size(dfm.x)))/minimum(size(dfm.x))
end


#  default criteria which are not quite consistent at least not when the number of factors is estimated TODO: why exactly is that?
function criterion_BIC(dfm::DynamicFactorModel)
    T, N = size(dfm.x)
    factor_residual_variance(dfm) + dfm.number_of_factors * log(T)/T
end
