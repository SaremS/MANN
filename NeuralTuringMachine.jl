#https://arxiv.org/pdf/1410.5401.pdf
#https://arxiv.org/pdf/1807.08518.pdf
using Flux
using LinearAlgebra
using Flux.Tracker: @grad
using Flux: hidden

mutable struct NTMCell

  N::Integer
  M::Integer

  input_layer::Chain
  output_layer::Chain

  memory
  write_array
  read_array

  write_head
  write_kt
  write_βt
  write_gt
  write_st
  write_γt

  write_at
  write_et

  read_head
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


  input_layer = Chain(Dense(in, hidden, relu))
  output_layer = Chain(Dense(hidden + N, out))

  memory = [TrackedArray(Float32.(ones((M)).*1e-6)) for n in 1:N]
  write_array = [TrackedArray(Float32.(softmax(randn((M))))) for n in 1:N]
  read_array = [TrackedArray(Float32.(softmax(randn((M))))) for n in 1:N]

  write_head = [Chain(Dense(hidden, M), softmax) for n in 1:N]
  write_kt = Dense(hidden, M, relu)
  write_βt = Dense(hidden, 1, relu)
  write_gt = Dense(hidden, 1, σ)
  write_st = [Chain(Dense(hidden, 3), softmax) for n in 1:N]
  write_γt = Chain(Dense(hidden, 1, relu), x->x.+1.)

  write_at = Dense(hidden, M)
  write_et = Dense(hidden, M, σ)

  read_head = [Chain(Dense(hidden, M), softmax) for n in 1:N]
  read_kt = Dense(hidden, M, relu)
  read_βt = Dense(hidden, 1, relu)
  read_gt = Dense(hidden, 1, σ)
  read_st = [Chain(Dense(hidden, 3), softmax) for n in 1:N]
  read_γt = Chain(Dense(hidden, 1, relu), x->x+1)


  cell = NTMCell(N,M,
                 input_layer, output_layer,
                 memory, write_array, read_array,
                 write_head, write_kt, write_βt, write_gt, write_st, write_γt,
                 write_at, write_et,
                 read_head, read_kt, read_βt, read_gt, read_st, read_γt)

  return cell
end





function content_focus(βt, kt, Mt)
  return softmax(βt*cosine_similarity(kt, Mt))
end

function cosine_similarity(u, v)
  return (dot(transpose(u),v))/(norm(u)*norm(v))
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
  padded_array = circular_padding(x)
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
  #return Mt1*(ones(size(wt)[1]) .- wt.*et)
  return ones(size(wt)[1]) .- wt.*et
end

function add_memory(Mtt, wt, at)
  return Mtt .+ wt .* at
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

  write_head = [m.write_head[i](latent) for i in 1:m.N]
  write_kt = m.write_kt(latent)
  write_βt = m.write_βt(latent)
  write_gt = m.write_gt(latent)
  write_st = [m.write_st[i](latent) for i in 1:m.N]
  write_γt = m.write_γt(latent)

  write_at = m.write_at(latent)
  write_et = m.write_kt(latent)

  read_head = [m.read_head[i](latent) for i in 1:m.N]
  read_kt = m.write_kt(latent)
  read_βt = m.write_βt(latent)
  read_gt = m.write_gt(latent)
  read_st = [m.write_st[i](latent) for i in 1:m.N]
  read_γt = m.write_γt(latent)

  read_array = [address_memory(read_βt, read_kt, m.memory[i], read_gt, m.read_array[i], read_st[i], read_γt) for i in 1:m.N]
  write_array = [address_memory(write_βt, write_kt, m.memory[i], write_gt, m.write_array[i], write_st[i], write_γt) for i in 1:m.N]



  rt = [read_memory(m.memory[i], read_array[i]) for i in 1:m.N]
  Mt = [write_memory(m.memory[i], write_array[i], write_et, write_at) for i in 1:m.N]

  #println("rt")
  #println(size((Tracker.collect(hcat(rt...)))[1,:]))
  #println(size(latent))

  latent_concat = cat(Tracker.collect(hcat(rt...))[1,:],latent;dims=1)
  #println(latent_concat)

  output = m.output_layer(latent_concat)

  return (Mt, write_array, read_array), output
end


function Flux.hidden(m::NTMCell)
  return (m.memory, m.write_array, m.read_array)
end



Flux.@treelike NTMCell

NTM(a...;ka...) = Flux.Recur(NTMCell(a...;ka...))



test = NTM(2, 2, 1, 3, 1)


test(input)
mem = test.memory
wra = test.write_array
rra = test.read_array

input = Float32.(ones((2)))

mem
test((mem, wra, rra), input)
println("\n")
