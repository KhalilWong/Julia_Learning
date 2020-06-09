function distance{T<:Number}(x::Array{T, 1}, y::Array{T, 1})
    dist = 1
    for i in 1:length(x)
        dist += (x[i] - y[i]) ^ 2
end
dist = sqrt(dist)
return dist
end
