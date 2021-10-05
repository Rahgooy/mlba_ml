rm(list=ls())

# Load the fit data from E2
extractFitParams = function(path){
  load(path)
  names = c('phi', 'phi.names', 'burnin', 'n.posteriors', 'nmc', 'n.chains')
  fitParams = list()
  for (p in names) {
    fitParams[[p]] <- get(p)
  }
  fitParams
}

newSeed=Sys.time()
set.seed(as.numeric(newSeed))


##### BAYES MODELLING STUFF
rdata=function(n,x,par.names,stim, conds) {
  names(x)=par.names
  A=x["A.mu"]
  s=c(1,1,1)
  out=list(Cond=NULL,Resp=NULL,Time=NULL)
  b=x["b.mu"] + x["A.mu"]
  t0=x["t0.mu"]
  I_0=x["I_0.mu"]
  m=x["m.mu"]
  lambda1=x["lambda1.mu"]
  lambda2=x["lambda2.mu"]
  gamma=x["gamma.mu"]
  beta=x["beta.mu"]
  
  for (cond in conds) {       			
    use.stim=stim[,,cond]
    v=getDrifts(stimuli=use.stim,I_0=I_0,m=m,lambda1=lambda1,lambda2=lambda2,gamma=gamma,beta=beta)
    
    
    tmp=rlba(n=n[cond],b=b,A=A,v=v,s=s,t0=t0)
    
    
    out$Resp=c(out$Resp,tmp$resp)
    out$Time=c(out$Time,tmp$rt)
    out$Cond=c(out$Cond,rep(cond,n[cond]))
    
  }
  out
}

##### Create a mesh gird
mesh_data=function(n, step) {
  x = rowMeans(fitParams$phi[1,,])
  names(x)=fitParams$phi.names
  
  #A= 1 
  #s=c(1,1,1)
  #b= 2 
  #t0=0 
  #I_0= 5
  #m= 5 
  #lambda1=0.2
  #lambda2=0.4
  #gamma= 1
  #beta=1
  #X = c(4., 6.)
  #Y = c(6., 4.)
  
  A=x["A.mu"]
  s=c(1,1,1)
  b=x["b.mu"] + x["A.mu"]
  t0=x["t0.mu"]
  I_0=x["I_0.mu"]
  m=x["m.mu"]
  lambda1=x["lambda1.mu"]
  lambda2=x["lambda2.mu"]
  gamma=x["gamma.mu"]
  beta=x["beta.mu"]
  
  X = c(3., 4.)
  Y = c(4., 3.)
  stim=array(NA,c(3,2))
  stim[1,] = X
  stim[2,] = Y
  mesh = list(C1=NULL, C2=NULL, p1=NULL, p2=NULL, p3=NULL, d1=NULL, d2=NULL, d3=NULL)
  
  for(x in seq(1, 6, by = step)){
    for(y in seq(1, 6, by = step)){
      stim[3,] = c(x, y)
      v=getDrifts(stimuli=stim,I_0=I_0,m=m,lambda1=lambda1,lambda2=lambda2,gamma=gamma,beta=beta)
      tmp=rlba(n=n,b=b,A=A,v=v,s=s,t0=t0)
      mesh$C1= c(mesh$C1, x)
      mesh$C2= c(mesh$C2, y)
      mesh$p1= c(mesh$p1, mean(tmp$resp == 1))
      mesh$p2= c(mesh$p2, mean(tmp$resp == 2))
      mesh$p3= c(mesh$p3, mean(tmp$resp == 3))
      mesh$d1= c(mesh$d1, v[1])
      mesh$d2= c(mesh$d2, v[2])
      mesh$d3= c(mesh$d3, v[3])
    }
    print(x)
  }
  mesh
}

########################################## load the functions you will use

source("lba-math.R")
source("driftFunctions.R")
require(msm)
library(jsonlite)

pred = function(path, nSamples) {
  # Load the experiment data for prediction
  cat("\n", "\n", "Making predictions for : ", path, "\n")
  load(path)
  pred=list()
  for (i in 1:S) {
    cat(i)
    tmpn=table(data[[i]]$Cond)
    totalN = sum(tmpn)
    tmpdat=list(Cond=NULL,Time=rep(0, totalN),Resp=list(O1=rep(0, totalN), O2=rep(0, totalN), O3=rep(0, totalN)))
    # Takes n.posteriors samples from posteriors and for each sample, predicts
    # the outcomes for all problems presented to each person
    par_m = 0
    par_A = 0
    par_chi = 0
    for (m in round(seq(from=fitParams$burnin,to=fitParams$nmc,length.out=nSamples))) {
      pars=fitParams$phi[sample(fitParams$n.chains,1),,m]
      par_A = par_A + pars[1]
      par_chi = par_chi + pars[3]
      par_m = par_m + pars[17]
      tmp=rdata(n=tmpn,x=pars,par.names=fitParams$phi.names,stim=stim[,,,i], conds=conds)
      tmpdat$Cond=tmp$Cond
      tmpdat$Time= tmpdat$Time + tmp$Time
      tmpdat$Resp$O1 = tmpdat$Resp$O1 + (tmp$Resp == 1)
      tmpdat$Resp$O2 = tmpdat$Resp$O2 + (tmp$Resp == 2)
      tmpdat$Resp$O3 = tmpdat$Resp$O3 + (tmp$Resp == 3)
    }
    print(par_m / nSamples)
    print(par_A / nSamples)
    print(par_chi / nSamples)
    tmpdat$Time = tmpdat$Time / nSamples
    tmpdat$Resp$O1 = tmpdat$Resp$O1 / nSamples
    tmpdat$Resp$O2 = tmpdat$Resp$O2 / nSamples
    tmpdat$Resp$O3 = tmpdat$Resp$O3 / nSamples
    pred[[i]]=data.frame(tmpdat)
  }
  names(pred)=names(data)
  pred
}
percent=30
fitParams = extractFitParams(paste("MLBA-E2-BayesHierFit-5000-", percent, "-percent.Rdata", sep=""))
e1a.pred = pred("../Data/parsedData-E1a.Rdata", 10000)
e1b.pred = pred("../Data/parsedData-E1b.Rdata", 10000)
e1c.pred = pred("../Data/parsedData-E1c.Rdata", 10000)

exportJSON <- toJSON(e1a.pred)
write(exportJSON, paste("e1a.", percent, "percent.json", sep = ""))

exportJSON <- toJSON(e1b.pred)
write(exportJSON, paste("e1b.", percent, "percent.json", sep = ""))

exportJSON <- toJSON(e1c.pred)
write(exportJSON, paste("e1c.", percent, "percent.json", sep = ""))

#mesh = mesh_data(1000, 0.05)  
#write.csv(mesh, 'mesh_E2.csv')

#savefile=paste("MLBA-E2-pred-E1",fitParams$nmc,sep="-")

#savefile=paste(savefile,"Rdata",sep=".")

#save.image(savefile)

