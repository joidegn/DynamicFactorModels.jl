using Base.Test
using DynamicFactorModels
using DataFrames

## Some test data
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
T, N = size(x)

# test residuals from factor equation
model = DynamicFactorModel(y,w,x)
sum_squared_residuals = sum(model.factor_residuals.^2)
