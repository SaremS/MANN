#https://arxiv.org/pdf/1410.5401.pdf
#https://arxiv.org/pdf/1807.08518.pdf
using Flux
using LinearAlgebra
using Flux.Tracker: @grad
using Flux: hidden
using Distributions

mutable struct NTMCell

  N::Integer
  M::Integer

  input_layer::Chain
  output_layer::Chain

  memory
  write_array
  read_array

  write_kt
  write_βt
  write_gt
  write_st
  write_γt

  write_at
  write_et

  
  read_kt
  read_βt
  read_gt
  read_st
  read_γt

end

M = 1
N = 3

ones((1,M)).*1e-6

function NTMCell(in::Integer, hidden::Integer, out::Integer, N::Integer, M::Integer)


  input_layer = Chain(Dense(in, hidden, selu))
  output_layer = Chain(Dense(hidden + M, hidden,selu), Dense(hidden, out))

  memory = [TrackedArray(Float32.(ones((M)).*1e-6)) for n in 1:N]
  write_array = TrackedArray(Float32.(softmax(randn((N)))))
  read_array = TrackedArray(Float32.(softmax(randn((N)))))

  write_kt = Dense(hidden, M, tanh)
  write_βt = Dense(hidden, 1, relu)
  write_gt = Dense(hidden, 1, σ)
  write_st = Chain(Dense(hidden, 3), softmax)
  write_γt = Chain(Dense(hidden, 1, relu), x->x.+1.)

  write_at = Dense(hidden, M, tanh)
  write_et = Dense(hidden, M, σ)

  read_kt = Dense(hidden, M, tanh)
  read_βt = Dense(hidden, 1, relu)
  read_gt = Dense(hidden, 1, σ)
  read_st = Chain(Dense(hidden, 3), softmax)
  read_γt = Chain(Dense(hidden, 1, relu), x->x.+1.)


  cell = NTMCell(N,M,
                 input_layer, output_layer,
                 memory, write_array, read_array,
                 write_kt, write_βt, write_gt, write_st, write_γt,
                 write_at, write_et,
                 read_kt, read_βt, read_gt, read_st, read_γt)

  return cell
end





function content_focus(βt, kt, Mt)
  cosine_s = x -> cosine_similarity(kt, x)
  exped = exp.(βt.*cosine_s.(Mt))
  exped = Tracker.collect(collect(Tracker.collect.(exped)))
  
  return exped ./ sum(exped)
end

function cosine_similarity(u, v)
  return sum(u .* v)/(norm(u)*norm(v))
end

function location_focus(gt, wtc, wt1)  
  return (gt.*wtc .+ (1. .-gt).*wt1)
end


function circular_padding(x::Array)
  return((cat(x[end], x, x[1]; dims=1)))
end

function circular_padding(x::TrackedArray)
  return(TrackedArray(cat(x[end].data, x.data, x[1].data; dims=1)))
end
@grad function circular_padding(x::TrackedArray)
  return(cat(x[end].grad, x.grad, x[1].grad; dims=1))
end


function circular_convolution(x, kernel)
  k = size(x)[1]
  padded_array = Tracker.collect(circular_padding(x))
  result = vcat([(reshape(padded_array[i:i+2],(1,3))*kernel) for i in 1:k]...)
  return result
end

function sharpen_operation(wt, γt)
  return wt.^γt ./ sum(wt .^ γt)
end

function address_memory(βt, kt, Mt, gt, wt1, st, γt)
  wtc = content_focus(βt, kt, Mt)
  wtg = location_focus(gt, wtc, wt1)
  wtt = circular_convolution(wtg, st)


  wt = sharpen_operation(wtt, γt)
  
  return wt
end

function erase_memory(Mt1, wt, et)

  result = [Mt1[i] .* (1 .- wt[i] .* et) for i in 1:size(Mt1)[1]]
  
  return result
end

function add_memory(Mtt, wt, at)

  result = [Mtt[i] .+ wt[i] .* at for i in 1:size(Mtt)[1]]
  
  return result
end

function write_memory(Mt1, wt, et, at)
  Mtt = erase_memory(Mt1, wt, et)
  Mt = add_memory(Mtt, wt, at)

  return Mt
end

function read_memory(Mt, wt)
  return wt .* Mt
end

function (m::NTMCell)((memory, write_array, read_array), x)

  latent = m.input_layer(x)

  write_kt = m.write_kt(latent)
  write_βt = m.write_βt(latent)
  write_gt = m.write_gt(latent)
  write_st = m.write_st(latent) 
  write_γt = m.write_γt(latent)

  write_at = m.write_at(latent)
  write_et = m.write_et(latent)

  read_kt = m.read_kt(latent)
  read_βt = m.read_βt(latent)
  read_gt = m.read_gt(latent)
  read_st = m.read_st(latent)
  read_γt = m.read_γt(latent)


  new_read_array = address_memory(read_βt, read_kt, memory, read_gt, read_array, read_st, read_γt)
  new_write_array = address_memory(write_βt, write_kt, memory, write_gt, write_array, write_st, write_γt)


  rt = read_memory(memory, new_read_array)
  Mt = write_memory(memory, new_write_array, write_et, write_at)

  rt = reduce(+, rt)


  latent_concat = cat(rt,latent;dims=1)

  output = m.output_layer(latent_concat)

  return (Mt, new_write_array, new_read_array), output
end


function Flux.hidden(m::NTMCell)
  return (m.memory, m.write_array, m.read_array)
end


Flux.@treelike NTMCell

NTM(a...;ka...) = Flux.Recur(NTMCell(a...;ka...))







X = Float32.(collect(range(0; stop = 5, length=100)))
y = Float32.(sin.(X*5.))
X = [[X[i]] for i in 1:100]


test2 = NTM(1,30, 1, 10, 10)

function loss(x,y)
  l = mean((Tracker.collect(vcat(test2.(x)...)) .- y).^2)
  println(l.data)
  Flux.truncate!(test2)
  Flux.reset!(test2)
  return l
end



opt = ADAM(0.1)


Flux.reset!(test2)
for i in 1:1500
  Flux.train!(loss, Flux.params(test2), [[X, y]], opt)
end

forecast = test2.(X)

Xd = Float32.(collect(range(5; stop = 10, length=100)))
yd = Float32.(sin.(Xd * 5.))
Xd = [[Xd[i]] for i in 1:100]

forecast2 = test2.(Xd)


using Plots
plot(vcat(Xd...), yd)
plot!(vcat(Xd...), [f.data[1] for f in forecast2])
