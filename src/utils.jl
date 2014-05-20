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
function flatten(arg, flat)
    iter_state = start(arg)
    if iter_state == false
        push!(flat, arg)  # arg cannot be flattened further
    else
        while !done(arg, iter_state)
            (item, iter_state) = next(arg, iter_state)
            flatten(item, flat)
        end
    end
    flat
end
flatten(args...) = apply(flatten, (args, Array(Any, 0)))
flatten(depth, args...) = apply(flatten, (args, Array(Any, 0)))


normalize(A::Matrix) = (A.-mean(A,1))./std(A,1) # normalize (i.e. center and rescale) Matrix A
normalize(A::Matrix, by) = (A.-by[1])./by[2] # normalize (i.e. center and rescale) Matrix A by given (mean, stddev)-tuple


norm_vector{T<:Number}(vec::Array{T, 1}) = vec./norm(vec) # makes vector unit norm
norm_matrix{T<:Number}(mat::Array{T, 2}) = mapslices(norm_vector, mat, 2)  # call norm_vector for each column

possemidef(x) = try 
    chol(x)
    return true
catch
    return false
end

function make_factor_model_design_matrix(w, factors, number_of_factors, number_of_factor_lags)
    return(hcat(w, factors))  # TODO: unfinished business
    design_matrix = hcat(w, factors)
    for lag_num in 1:number_of_factor_lags
        lag_matrix = 1
    end
end

function pseudo_out_of_sample_forecasts(model, y, w, x, model_args...; num_predictions::Int=200)
    # one step ahead pseudo out-of-sample forecasts
    T = length(y)
    predictions = zeros(num_predictions)
    true_values = zeros(num_predictions)
    for date_index in T-num_predictions+1:T
        month = date_index - T+num_predictions
        println("month: $month")
        let y=y[1:date_index], x=x[1:date_index, :], w=w[1:date_index, :]  # y, x and w are updated so its easier for humans to read the next lines
            let newx=x[end, :], neww=w[end, :], newy=y[end], y=y[1:end-1], x=x[1:end-1, :], w=w[1:end-1, :]  # pseudo-one step ahead (keeps notation clean in the following lines)
                args = length(model_args) == 0 ? (y, w, x) : tuple(y, w, x, model_args...)  # efficiency concerns?
                res = apply(model, args)
                predictions[month] = (predict(res, neww, newx))[1]
                true_values[month] = newy
            end
        end
    end
    return(predictions, true_values)
end

MSE(y, predictions) = sum((y-predictions).^2)/apply(*, size(y))

