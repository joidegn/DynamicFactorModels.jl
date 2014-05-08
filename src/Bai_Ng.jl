# trying to replicate Bai and Ng 2002

using DynamicFactorModels
using DimensionalityReduction

@time results1 = table1(); gc()
@time results2 = table2(); gc()
@time results3 = table3(); gc()
results = vcat(results1, results2, results3)

function table1()  # replicate main information criteria of table 1
    tuples = [(100, 100, 1), (100, 40, 1), (100, 60, 1), (200, 60, 1), (500, 60, 1), (1000, 60, 1), (2000, 60, 1), (4000, 60, 1), (4000, 100, 1), (8000, 60, 1), (8000, 100, 1), (50, 10, 1), (100, 10, 1), (100, 20, 1)]  # T, N, r combinations I can replicate
    criteria = ["PCp1", "PCp2", "PCp3", "ICp1", "ICp2", "ICp3"]
    results1 = zeros(length(tuples), length(criteria))
    for criterion_idx in 1:length(criteria)
        criterion = criteria[criterion_idx]
        println("crunching criterion $criterion")
        for i in 1:length(tuples)
            println("tuple: ", tuples[i])
            y, x, f, lambda, epsilon_x, epsilon_y = apply(factor_model_DGP, tuples[i])
            x = normalize(x)
            w = reshape(ones(length(y)), (length(y), 1))  # reshape to Array{Float64, 2}
            try 
                dfm = DynamicFactorModel(y, w, x, criterion)
                results1[i, criterion_idx] = dfm.number_of_factors
            catch exception
                println("caught error. Probably because of negative variances.")
                println(exception)
            end
        end
    end
    results1
end
function table2()
    tuples = [(100, 100, 3), (100, 40, 3), (100, 60, 3), (200, 60, 3), (500, 60, 3), (1000, 60, 3), (2000, 60, 3), (4000, 60, 3), (4000, 100, 3), (8000, 60, 3), (8000, 100, 3), (50, 10, 3), (100, 10, 3), (100, 20, 3)]  # T, N, r combinations I can replicate
    criteria = ["PCp1", "PCp2", "PCp3", "ICp1", "ICp2", "ICp3"]
    results2 = zeros(length(tuples), length(criteria))
    for criterion_idx in 1:length(criteria)
        criterion = criteria[criterion_idx]
        println("crunching criterion $criterion")
        for i in 1:length(tuples)
            println("tuple: ", tuples[i])
            y, x, f, lambda, epsilon_x, epsilon_y = apply(factor_model_DGP, tuples[i])
            x = normalize(x)
            w = reshape(ones(length(y)), (length(y), 1))  # reshape to Array{Float64, 2}
            try
                dfm = DynamicFactorModel(y, w, x, criterion)
                results2[i, criterion_idx] = dfm.number_of_factors
            catch exception
                println("caught error. probably due to negative variances")
                println(exception)
            end
        end
    end
    results2
end
function table3()
    tuples = [(100, 100, 5), (100, 40, 5), (100, 60, 5), (200, 60, 5), (500, 60, 5), (1000, 60, 5), (2000, 60, 5), (4000, 60, 5), (4000, 100, 5), (8000, 60, 5), (8000, 100, 5), (50, 10, 5), (100, 10, 5), (100, 20, 5)]  # T, N, r combinations I can replicate
    criteria = ["PCp1", "PCp2", "PCp3", "ICp1", "ICp2", "ICp3"]
    results3 = zeros(length(tuples), length(criteria))
    for criterion_idx in 1:length(criteria)
        criterion = criteria[criterion_idx]
        println("crunching criterion $criterion")
        for i in 1:length(tuples)
            println("tuple: ", tuples[i])
            y, x, f, lambda, epsilon_x, epsilon_y = apply(factor_model_DGP, tuples[i])
            x = normalize(x)
            w = reshape(ones(length(y)), (length(y), 1))  # reshape to Array{Float64, 2}
            try
                dfm = DynamicFactorModel(y, w, x, criterion)
                results3[i, criterion_idx] = dfm.number_of_factors
            catch exception
                println("caught errror. probably due to negative variances.")
                println(exception)
            end
        end
    end
    results3
end
