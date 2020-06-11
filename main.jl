using Random

################################################################################
function distance(x::Array{T, 1}, y::Array{T, 1}) where {T<:Number}
    dist = 0
    for i in 1:length(x)
        dist += (x[i] - y[i]) ^ 2
    end
    dist = sqrt(dist)
    return dist
end
################################################################################
function classify(distances::Array{Float64, 1}, labels::Array{T, 1}, k::Int64) where {T<:Any}
    class = unique(labels)                                                      #所有类别不重复
    nc = length(class)                                                          #类别数
    indexes = Array{Int}(undef, k)                                                     #
    M = typemax(typeof(distances[1]))
    class_count = Array{Int}(undef, nc)
    for i in 1:k
        d_min, indexes[i] = findmin(distances)
        distances[indexes[i]] = M
    end
    klabels = labels[indexes]
    for i in 1:nc
        for j in 1:k
            if klabels[j] == class[i]
                class_count[i] += 1
                break
            end
        end
    end
    d_max, index = findmax(class_count)
    return class[index]
end
################################################################################
function apply_kNN(X::Array{T1, 2}, x::Array{T2, 1}, Y::Array{T1, 2}, k::Int) where {T1<:Number, T2<:Any}
    N = size(X, 1)                                                              #已知数据点
    n = size(Y, 1)                                                              #待分类数据点
    D = Array{Float64}(undef, N)                                                       #初始化距离矢量（中间计算距离过程）
    z = Array{typeof(x[1])}(undef, n)                                                  #初始化标签矢量（最终分类结果）
    for i in 1:n
        for j in 1:N
            D[j] = distance(X[j, :], Y[i, :])
        end
        z[i] = classify(D, x, k)
    end
    return z
end
################################################################################
println("\nBeginning...")
println(pwd())
#data = readcsv("D:\\data\\magic04.data");                                        #分号-避免数据显示在窗口
#
f = open("magic04.data", "r")
lines = readlines(f);
close(f)
N = length(lines)
FN = length(split(lines[1], ","))
println(FN)
#FN = parse(Float64, lineString[1])
#println(typeof(FN))
#
I = Array{Float64, 2}(undef, N, FN - 1)
O = Array{Any}(undef, N)
for i in 1:N
    lineString = split(lines[i], ",")
    for j in 1:FN - 1
        I[i, j] = parse(Float64, lineString[j])
    end
    O[i] = lineString[FN]
end
println(typeof(O))
println(typeof(O[1]))
n = round(Int64, N / 3 * 2)
R = randperm(N)                                                                 #随机排列
indX = R[1:n]
X = I[indX, :]
x = O[indX]
indY = R[(n + 1):end]
Y = I[indY, :]
y = O[indY]
z = apply_kNN(X, x, Y, 5)
println(z[1:5])
println(y[1:5])
println(sum(y[1:5] .== z[1:5]))
println(sum(y .== z) / n)
#println(z[1:5])
#println(z[1:5])
println("End\n")
