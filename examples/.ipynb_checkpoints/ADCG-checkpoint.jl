export ADCG
using Printf

function ADCG(sim :: ForwardModel, lossFn :: Loss, y :: Vector{Float64}, tau :: Float64;
  callback :: Function = (old_thetas,thetas, weights,output,old_obj_val) -> false,
  max_iters :: Int64 = 50,
  min_optimality_gap :: Float64 = 1E-5,
  max_cd_iters :: Int64 = 200,
  fully_corrective :: Bool = false)
  #assert(tau > 0.0)
  if (tau<0.0)
    println("The maximal mass of the total measure (tau) cannot be less than zero")
  end
  bound = -Inf
  thetas = zeros(0,0) #hack
  weights = zeros(0)
  #cache the forward model applied to the current measure.
  output = zeros(length(y))
  for iter = 1:max_iters
    #compute the current residual
    residual = output - y
    #evalute the objective value and gradient of the loss
    objective_value, grad = loss(lossFn, residual)
    #compute the next parameter value to add to the support
    theta,score = lmo(sim,grad)
    #println("The objective value for the current residual is: ", string(objective_value))
    #score is - |<\psi(theta), gradient>|
    #update the lower bound on the optimal value
    bound = max(bound, objective_value+score*tau-dot(output,grad))
    #println("The lower bound on the optimal value is: ", string(bound))
    #check if the bound is met.
    #if(objective_value > bound + min_optimality_gap || score >= 0.0)
      #print('h', thetas)
    #  return thetas,weights
    #end
    #update the support
    old_thetas = thetas
    thetas = iter == 1 ? reshape(theta, length(theta),1) : [thetas theta]
    #run local optimization over the support.
    old_weights = copy(weights)
    thetas,weights = localUpdate(sim,lossFn,thetas,y,tau,max_cd_iters)
    output = phi(sim, thetas, weights)
    if callback(old_thetas, thetas,weights, output, objective_value)
      return old_thetas, old_weights
    end
  end
  println("Hit max iters in frank-wolfe!")
  return thetas, weights
end

#Default implementation using coordinate descent
#Feel free to override
function localUpdate(sim :: ForwardModel,lossFn :: Loss,
    thetas :: Array{Float64, 2}, y :: Array{Float64,1}, tau :: Float64, max_iters)
  for cd_iter = 1:max_iters
    weights = solveFiniteDimProblem(sim, lossFn, thetas, y, tau)
    #remove points with zero weight
    if any(weights.==0.0)
      println("Removing ",sum(weights.==0.0), " zero-weight points.")
      thetas = thetas[:,weights.!= 0.0]
      weights = weights[weights.!= 0.0]
    end
    #local minimization over the support
    new_thetas = localDescent(sim, lossFn, thetas,weights, y)
    #break if termination condition is met
    if length(thetas) == length(new_thetas) && maximum(abs.(vec(thetas)-vec(new_thetas))) <= 1E-7
        break
    end
    thetas = new_thetas
  end
  #final prune
  weights = solveFiniteDimProblem(sim, lossFn, thetas, y, tau)
  if any(weights.==0.0)
    println("Removing ",sum(weights.==0.0), " zero-weight points.")
    thetas = thetas[:,weights.!= 0.0]
    weights = weights[weights.!= 0.0]
  end
  return thetas, weights
end
