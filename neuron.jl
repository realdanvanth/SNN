using Random
using PyPlot
mutable struct Node
    w1::Float64
    b1::Float64
    w2::Float64
end
function activation( w::Float64,b::Float64,x::Float64)
  return log(1+exp(w*x+b))
end
function dactivation(w::Float64,b::Float64,x::Float64)
  return exp(w*x+b)/(1+exp(w*x+b))
end
function predict(nodes::Vector{Node},fb::Float64,x::Float64)
  result=0.0
  for node in nodes
    result+=activation(node.w1,node.b1,x)*node.w2
  end
  result+=fb
  return result
end
function dMSE(nodes::Vector{Node},fb::Float64)
  sum=0.0
  for i in -1.0:0.1:1.0
    sum+=(data(i)-predict(nodes,fb,i))
  end
  return -2*sum
end

function gradientdescent(nodes::Vector{Node},fb::Float64,lr::Float64)
  newnodes=Array{Node, 1}(undef,length(nodes))
  for i in 1:length(nodes)
    newnodes[i]=Node(0.0,0.0,0.0)
  end
  fb-=dMSE(nodes,fb)*lr
  for i in eachindex(nodes)
    dw1=0.0
    db1=0.0
    dw2=0.0
    for j in -1:0.1:1.0
      error=data(j)-predict(nodes,fb,j)
      dw2+=-2*(error)*activation(nodes[i].w1,nodes[i].b1,j)
      db1+=-2*(error)*dactivation(nodes[i].w1,nodes[i].b1,j)
      dw1+=-2*(error)*dactivation(nodes[i].w1,nodes[i].b1,j)*j
    end
    newnodes[i].w2=nodes[i].w2-dw2*lr
    newnodes[i].w1=nodes[i].w1-dw1*lr
    newnodes[i].b1=nodes[i].b1-db1*lr
  end
  return newnodes,fb
end
function data(x::Float64)
  return sin(x)
end
function MSE(nodes::Vector{Node},fb::Float64)
  sum=0.0
  for i in -1.0:0.1:1.0 
    sum+=(data(i)-predict(nodes,fb,i))^2
  end
  return sum
end
function main()
  n = 2
  nodes = Array{Node, 1}(undef,n)
  for i in 1:n
    nodes[i]=Node(rand(),0.0,randn())
  end
  fb=0.0
  lr=0.01
  epochs=50
  x = collect(-1.0:0.1:1.0) 
  y= [predict(nodes, fb, xi) for xi in x]
  imse=MSE(nodes,fb)
  println("Initial MSE: ", imse)
  println("Prediction for 0.1: ",predict(nodes,fb,0.1)," sin(0.1): ",sin(0.1))
  for i in 1:epochs
    nodes,fb=gradientdescent(nodes,fb,lr)
    imse=MSE(nodes,fb)
    #println("epoch $i MSE: ", imse)
  end
  for i in eachindex(nodes)
    println("Node $i: w1 = ", nodes[i].w1, ", b1 = ", nodes[i].b1, ", w2 = ", nodes[i].w2)
  end 
  println("Prediction for 0.1: ",predict(nodes,fb,0.1)," sin(0.1): ",sin(0.1))
  newy= [predict(nodes, fb, xi) for xi in x]
  plot(x, sin.(x), label="sin(x)",color="blue")
  plot(x, y, label="Initial", linestyle="dashed",color="red") 
  plot(x, newy, label="Final", linestyle="dashed",color="green")
  show()
end
main()

