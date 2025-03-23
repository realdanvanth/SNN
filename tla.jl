using Random
using PyPlot
mutable struct Layer1
    w::Float64
    b::Float64
    weights::Vector{Float64}
end
mutable struct Layer2
    w::Float64#future weights
    b::Float64#neuron bias
end
function activation( w::Float64,b::Float64,x::Float64)
 return  max(1.0 - abs(w*x+b), 0.0) 
end
function dactivation(w::Float64,b::Float64,x::Float64)
  if(w*x+b>-1.0&&w*x+b<0.0)
    return 1.0
  elseif (w*x+b>0.0&&w*x+b<1.0)
      return -1.0
  else
    return 0.0
  end
end
function activation(prevnodes::Vector{Layer1},node::Layer2,i::Int64,x::Float64)
  total=0.0
  for prevnode in prevnodes
    total+=(prevnode.weights[i]*activation(prevnode.w,prevnode.b,x))
  end
  return max(1.0 - abs(total+node.b), 0.0) 
end
function dactivation(prevnodes::Vector{Layer1},node::Layer2,i::Int64,x::Float64)
  total=0.0
  for prevnode in prevnodes
    total+=(prevnode.weights[i]*activation(prevnode.w,prevnode.b,x))
  end
  if(total+node.b>-1.0&&total+node.b<0.0)
    return 1.0
  elseif (total+node.b>0.0&&total+node.b<1.0)
    return -1.0
  else
    return 0.0
  end
end
function predict(nodes1::Vector{Layer1},nodes2::Vector{Layer2},fb::Float64,x::Float64)
  total=0.0
  for i in eachindex(nodes2)
    total+=(activation(nodes1,nodes2[i],i,x)*nodes2[i].w)
  end
  total+=fb
  return total
end
function data(x::Float64)
  return sin(x)^2+cos(x^2)
end
function dMSE(nodes1::Vector{Layer1},nodes2::Vector{Layer2},fb::Float64,s,step,e)
  sum=0.0
  for i in s:step:e
    sum+=(data(i)-predict(nodes1,nodes2,fb,i))
  end
  return -2*(sum)
end
function gradientdescent(nodes1::Vector{Layer1},nodes2::Vector{Layer2},fb::Float64,lr::Float64,s,step,e)
   fb-=dMSE(nodes1,nodes2,fb,s,step,e)/(((e - s) / step))*lr
   newnodes1 = deepcopy(nodes1)
   newnodes2 = deepcopy(nodes2)
   for i in eachindex(nodes2)
     dw=0.0
     db=0.0
     for j in s:step:e
       error=data(j)-predict(nodes1,nodes2,fb,j)
       dw += error*activation(nodes1,nodes2[i],i,j)
       db += error*activation(nodes1,nodes2[i],i,j)*dactivation(nodes1,nodes2[i],i,j)
     end
     newnodes2[i].w-=(-2*dw*lr)
     newnodes2[i].b-=(-2*db*lr)
   end
   for i in eachindex(nodes1)
     dw=0.0
     db=0.0
     for j in eachindex(nodes1[i].weights)
       dwj=0.0
       for x in s:step:e
         error=data(x)-predict(nodes1,nodes2,fb,x)
         act1 = activation(nodes1[i].w, nodes1[i].b, x)
         dact1 = dactivation(nodes1[i].w, nodes1[i].b, x)
         act2 = activation(nodes1,nodes2[j],i,x)
         dact2 = dactivation(nodes1,nodes2[j],i,x)
         dwj+=error*act2*dact2*act1
         db+=error*act2*dact2*act1*dact1
         dw+=error*act2*dact2*act1*dact1*x
        end
        newnodes1[i].weights[j]-=(-2*dwj*lr)
     end
     newnodes1[i].w-=(-2*dw*lr)
     newnodes1[i].b-=(-2*db*lr)
   end
   return newnodes1, newnodes2, fb
end
function main()
  n = 32 
  nodes1 = Array{Layer1, 1}(undef,n)
  nodes2 = Array{Layer2,1}(undef,n)
  for i in 1:n
    nodes1[i]=Layer1(randn()* sqrt(2.0 / n),0.0,randn(n)[:]* sqrt(2.0 / n))
  end
  w=pi
  for i in 1:n
    nodes2[i]=Layer2(randn()* sqrt(2.0 / n),0.0)
  end
  fb=0.0
  lr=0.001
  s=-pi
  e=pi
  step=0.5
  epochs=10000
  x = collect(s:step:e)
  y= [predict(nodes1,nodes2, fb, xi) for xi in x]
  for i in 1:epochs
    nodes1,nodes2,fb=gradientdescent(nodes1,nodes2,fb,lr,s,step,e)
    clf()
    newy = [predict(nodes1,nodes2, fb, xi) for xi in x]
    plot(x, data.(x), label="sin(x)", color="blue", linewidth=2)
    plot(x, y, label="Initial", linestyle="dashed", color="red", linewidth=2)
    plot(x, newy, label="Epoch $i", linestyle="dashed", color="green", linewidth=2)  
    legend()
    title("Neural Network Training Progress")
    xlabel("x")
    ylabel("y")
    grid(true)
    pause(0.001)
  end
  show()
end
main()
