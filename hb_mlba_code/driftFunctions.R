
getDrifts=function(stimuli, I_0, m, lambda1, lambda2 , gamma , beta) {
  
  # Calculates the mean drift rates for MLBA
  # stimuli is a 3x2 matrix with the attributes of each option
  
  # Beta seems to be a multiplier that is applied to the second attribute to allow it to differ from the first
  # Gamma seems to be a multiplier to transform things to "drift scale"
  
  d=rep(NA,3)
  
  u1 = CurveFun(stimuli[1,1:2], m)
  u2 = CurveFun(stimuli[2,1:2], m)
  u3 = CurveFun(stimuli[3,1:2], m)
  
  d[1] = gamma*Valuation(u1, u2, lambda1, lambda2 , beta) + gamma*Valuation(u1, u3, lambda1, lambda2 , beta) + I_0
  d[2] = gamma*Valuation(u2, u1, lambda1, lambda2 , beta) + gamma*Valuation(u2, u3, lambda1, lambda2 , beta) + I_0
  d[3] = gamma*Valuation(u3, u1, lambda1, lambda2 , beta) + gamma*Valuation(u3, u2, lambda1, lambda2 , beta) + I_0
  
  if (sum(is.na(d)) > 0) return(-Inf)
  
  if (max(d) < 0) {
    #stop("Highest drift is less than 0")
    d=rep(0,3)
  }
  
  d
}

CurveFun=function(option, m) {
  
  u=rep(NA,2)
  
  # Mapping objective to subjective values (Appendix C)
  
  # x and y intercepts of the line of indifference, which is in the direction of the unit-length vector 1/sqrt(2)*[-1,1]'
  a = option[1] + option[2]
  b = a
  # angle between x-axis and the vector <option(1), option(2)>
  angle = atan(option[2]/option[1])     
  # subjective values for the option on the curve (x/a)^m + (y/b)^m
  u[1] = b/((tan(angle))^m+(b/a)^m)^(1/m)     
  u[2] = b*(1-(u[1]/a)^m)^(1/m)
  u
}

Weight1=function(A, B, lambda1, lambda2) {
  # Attention weights (equation 4)
  
  if (A >= B) {
    wt =exp(-lambda1*abs(A-B))
  } else {
    wt =exp(-lambda2*abs(A-B))
  }
  wt
}

Weight2=function(A, B, lambda1, lambda2 , beta) {
# Attention weights (equation 4)

  if (A >= B) {
    wt =exp(-beta*lambda1*abs(A-B))
  } else {
    wt =exp(-beta*lambda2*abs(A-B))
  }
  wt
}

Valuation=function(option1, option2, lambda1, lambda2 , beta) {
# Valuation function (equation 3)
  
  v = Weight1(option1[1], option2[1], lambda1, lambda2)*(option1[1] - option2[1]) + Weight2(option1[2], option2[2], lambda1, lambda2 , beta)*(option1[2] - option2[2])
  v
}
