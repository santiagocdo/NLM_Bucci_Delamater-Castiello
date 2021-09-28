# PENDING. Andy Delamater, 02/09/2021
# 1. 6 non-reinforced instead of 2
# 2. Less nBlock 150 -> 75
# 3. mutiple phases (need v6)

# Remove all of the elements currently loaded in R
rm(list=ls(all=TRUE)) 



# functions
f_backProp <- function (param,training,nKO.MM_one) {
  
  a <- param$a
  b <- param$b
  nHid <- param$nHidden$nHV + param$nHidden$nHMM + param$nHidden$nHA
  nBlock <- param$nBlock
  actFun <- param$actFun
  INPUT <- training$INPUT
  OUTPUT <- training$OUTPUT
  TEST <- training$TEST
  trialType <- training$trialType
  
  ### create matrices and scalars
  nOut <- ncol(OUTPUT)
  
  nStim <- ncol(INPUT)
  
  nTrialType <- nrow(INPUT)
  
  nTest <- nrow(TEST)
  
  nTrial <- nTrialType*nBlock
  
  
  actOT <- matrix(NA,nrow = nTrialType, ncol = nOut)
  
  actH <- matrix(NA,nrow = nTrial, ncol = nHid)
  actO <- matrix(NA,nrow = nTrial, ncol = nOut)
  
  deltaH <- matrix(NA,nrow = nTrial, ncol = nHid)
  deltaO <- matrix(NA,nrow = nTrial, ncol = nOut)
  
  dwIH <- matrix(0, nrow = nStim, ncol = nHid)
  wIH <- matrix(runif(nStim*nHid)-0.5,nrow = nStim, ncol = nHid) # rows input units (stim), cols hidden units
  # remove connections
  conIH <- matrix(1, nrow = nStim, ncol = nHid)
  # block visual input with auditory hidden
  if (param$nHidden$nHA > 0) {
    conIH[(param$nInput$ctx+1):(param$nInput$ctx + param$nInput$vis),
          (nHid-param$nHidden$nHA + 1):nHid] <- 0}
  # block auditory input with visual hidden
  if (param$nHidden$nHV > 0) {
    conIH[(param$nInput$ctx + param$nInput$vis + 1):nStim,
          1:param$nHidden$nHV] <- 0}
  wIH <- wIH * conIH
  
  dwHO <- matrix(0, nrow = nHid, ncol = nOut)
  wHO <- matrix(runif(nHid*nOut)-0.5, nrow = nHid) # rows hidden units, cols output units or options
  
  
  #### backprop algorithm ###
  # see also: https://web.stanford.edu/group/pdplab/pdphandbook/handbookch6.html
  for (nB in 1:nBlock) {
    # randomized trials
    randT <- sample(1:nTrialType)
    INPUT <- INPUT[randT,]
    OUTPUT <- as.matrix(OUTPUT[randT,])
    trialType <- trialType[randT]
    for (t in 1:nTrialType) {
      
      ### ### activation ### ### forward activation ## chain rule: act_fun(act_fun(trialBlock %*% wIH) %*% wHO)
      trial <- t+((nB-1)*nTrialType)
      
      actH[trial,] <- actFun(INPUT[t,] %*% wIH) # INPUT[t,] assumed to be a 1 x nStim matrix
      
      actO[trial,] <- actFun(actH[trial,] %*% wHO) # actH[t,] assumed to be a 1 x nOut matrix
      
      
      
      ### ### learning (weight changes) ### ###
      
      ### delta (error) for output
      deltaO[trial,] <- (OUTPUT[t,] - actO[trial,]) * (actO[trial,] * (1 - actO[trial,]))
      
      ### delta (error) for hidden
      deltaH[trial,] <- (deltaO[trial,] %*% t(wHO[,])) * (actH[trial,] * (1 - actH[trial,])) #rowSums(as.matrix(deltaO[t,] * wHO[,]))
      
      
      # weight change Hidden-Output
      dwHO[,] <- t(a*(as.matrix(deltaO[trial,]) %*% actH[trial,])) + b*dwHO[,]
      
      # update weights Hidden-Output
      wHO[,] <- wHO[,] + dwHO[,]
      
      
      # weight change Input-Hidden
      dwIH[,] <- t(a*(as.matrix(deltaH[trial,]) %*% INPUT[t,])) + b*dwIH[,]
      
      # update weights Input-Hidden
      wIH[,] <- wIH[,] + (dwIH[,] * conIH)
      
    } # end trials per block cycle # t
    if (nB == 1) {
      trialTypeLong <- trialType
    } else {
      trialTypeLong <- c(trialTypeLong,trialType)
    }
  } # end all blocks cycle # nB
  
  ### test input patterns ###
  if (nKO.MM_one == 0) {
    for (t in 1:nTest) {
      actOT[t,] <- actFun(actFun(TEST[t,] %*% wIH) %*% wHO)
    }
  } else {
    KO_units <- sample(param$nHidden$nHV:(nHid-param$nHidden$nHA))[1:nKO.MM_one]
    for (t in 1:nTest) {
      wIH_KO <- wIH; wIH_KO[,KO_units] <- 0
      wHO_KO <- wHO; wHO_KO[KO_units,] <- 0
        
      actOT[t,] <- actFun(actFun(TEST[t,] %*% wIH_KO) %*% wHO_KO)
    }
  }
  
  
  db <- data.frame(nTrial=1:nTrial,nBlock=rep(1:nBlock,each=nTrialType),
                   trialType=trialTypeLong,
                   actO=actO)
  
  ouput <- list(db=db,actO=actO, wIH=wIH, wHO=wHO, actOT=actOT)
  
  return(ouput)
  
}
f_run_sims <- function (param,training,nSim,nKO.MM_one,KOdich) {
  for (s in 1:nSim) {
    # message(paste("",s))
    if (KOdich == 1) {
      temp <- f_backProp(param,training,nKO.MM_one)
      temp2 <- data.frame(nHMM=nKO.MM_one,nSim=s,
                          trialType=training$trialType,actTest=temp$actOT)
      temp <- data.frame(nHMM=nKO.MM_one,nSim=s,
                         melt(temp$db,id.vars = c("nTrial","nBlock","trialType")))
    } else {
      temp <- f_backProp(param,training,nKO.MM_one)
      temp2 <- data.frame(nHMM=param$nHidden$nHMM,nSim=s,
                          trialType=training$trialType,actTest=temp$actOT)
      temp <- data.frame(nHMM=param$nHidden$nHMM,nSim=s,
                         melt(temp$db,id.vars = c("nTrial","nBlock","trialType")))
    }
    
    if (s == 1) {
      actO <- temp
      actT <- temp2
    } else {
      actO <- rbind(actO,temp)
      actT <- rbind(actT,temp2)
    }
  } # end s loop
  return(list(actO_out=actO,actT_out=actT))
}
if (!require(reshape2)) {install.packages("reshape2")}; library(reshape2)
if (!require(ggplot2)) {install.packages("ggplot2")}; library(ggplot2)
if (!require(ggpubr)) {install.packages("ggpubr")}; library(ggpubr)
if (!require(viridis)) {install.packages("viridis")}; library(viridis)
if (!require(dplyr)) {install.packages("dplyr")}; library(dplyr)



######################### Figure 2 #############################################
############################################################################## #
############################################################################## #
# parameters
param <- list(a=0.3,b=0.9,nBlock=80,
              nHidden=data.frame(nHV=2,nHMM=4,nHA=2),
              nInput=data.frame(ctx=1,vis=1,aud=1),
              actFun=function (netin) {1/(1+exp(-(netin-2.2)))}
)

nTrTy <- 10
training <- list(
  INPUT = t(matrix(c(1,  1,  0, # V+
                     1,  0,  1, # A+
                     1,  1,  1, # VA-
                     1,  1,  1, # VA-
                     1,  0,  0, # ctx
                     1,  0,  0, # ctx
                     1,  0,  0, # ctx
                     1,  0,  0, # ctx
                     1,  0,  0, # ctx
                     1,  0,  0),nrow=3,ncol=10)),
  # INPUT = t(matrix(c(1,  1,  0, # V+
  #                    1,  0,  1, # A+
  #                    1,  1,  1, # VA-
  #                    1,  1,  1, # VA-
  #                    1,  1,  1, # VA-
  #                    1,  1,  1, # VA-
  #                    1,  1,  1, # VA-
  #                    1,  1,  1, # VA-
  #                    1,  0,  0,
  #                    1,  0,  0),nrow=3,ncol=nTrTy)),
  #                  O1  O2
  OUTPUT = t(matrix(c(1,  0, # V+
                      0,  1, # A+
                      0,  0, # VA-
                      0,  0, # VA-
                      0,  0, # ctx
                      0,  0, # ctx
                      0,  0, # ctx
                      0,  0, # ctx
                      0,  0, # ctx
                      0,  0),nrow=2,ncol=10)),
  # OUTPUT = t(matrix(c(1,  0, # V+
  #                     0,  1, # A+
  #                     0,  0, # VA-
  #                     0,  0, # VA-
  #                     0,  0, # VA-
  #                     0,  0, # VA-
  #                     0,  0, # VA-
  #                     0,  0, # VA-
  #                     0,  0,
  #                     0,  0),nrow=2,ncol=nTrTy)),
  #               cxt   V   A
  TEST = t(matrix(c(1,  1,  0, # V+
                    1,  0,  1, # A+
                    1,  1,  1, # VA-
                    1,  1,  1, # VA-
                    1,  0,  0, # ctx
                    1,  0,  0, # ctx
                    1,  0,  0, # ctx
                    1,  0,  0, # ctx
                    1,  0,  0, # ctx
                    1,  0,  0),nrow=3,ncol=10)),
  # TEST = t(matrix(c(1,  1,  0, # V+
  #                   1,  0,  1, # A+
  #                   1,  1,  1, # VA-
  #                   1,  1,  1, # VA-
  #                   1,  1,  1, # VA-
  #                   1,  1,  1, # VA-
  #                   1,  1,  1, # VA-
  #                   1,  1,  1, # VA-
  #                   1,  0,  0,
  #                   1,  0,  0),nrow=3,ncol=nTrTy)),
  trialType = c("V+","A+","VA-","VA-","ctx","ctx","ctx","ctx","ctx","ctx")
  # trialType = c("V+","A+","VA-","VA-","VA-","VA-","VA-","VA-","ctx","ctx")
)

nSim <- 32



plotsD <- list()
for (i in 1:nSim) {
  temp <- f_backProp(param,training,nKO.MM_one=0)
  temp$db$nBlock <- rep(1:(max(temp$db$nBlock)/10),each=100)
  if (i == 1) {
    dat2 <- data.frame(nSim = i,temp$db)
  } else {
    dat2 <- rbind(dat2,data.frame(nSim = i,temp$db))
  }
  
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  # algorithm to detect configural units
  for (j in 1:4) {
    wHO_KO <- temp$wHO; wHO_KO[c(1:2,7:8),] <- 0
    temp2 <- data.frame(group="D",nSim=i,
      trialType=training$trialType[c(1:3,5)[j]],
      actComp=param$actFun(param$actFun(training$INPUT[c(1:3,5)[j],] %*% temp$wIH) %*% temp$wHO),
      actKO=param$actFun(param$actFun(training$INPUT[c(1:3,5)[j],] %*% temp$wIH) %*% wHO_KO))
    if (j == 1) {
      tempKO <- temp2
    } else {
      tempKO <- rbind(tempKO,temp2)
    }
  }; remove(temp2,j)
  tempKO$actCompMinKO.1 <- tempKO$actComp.1 - tempKO$actKO.1
  tempKO$actCompMinKO.2 <- tempKO$actComp.2 - tempKO$actKO.2
  tempKO <- melt(tempKO, id.vars = c("group","nSim","trialType","actComp.1","actComp.2","actKO.1","actKO.2"))
  tempKO$variable <- as.character(tempKO$variable) 
  tempKO$outcome <- substr(tempKO$variable,nchar(tempKO$variable),nchar(tempKO$variable))
  tempKO$trialType <- factor(tempKO$trialType, levels = c("V+","A+","VA-","ctx"))
  levels(tempKO$trialType ) <- c("V+","A*","VA-","ctx")
  tempCor <- data.frame(group="D",nSim=i,cor=cor(temp$wIH[2,3:6],temp$wIH[3,3:6]))
  
  
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  outLabs <-  c("O1 (+)","O2 (*)"); names(outLabs) <- c("1","2")
  pActDiff <- ggplot(tempKO,aes(x=trialType,y=value,fill=trialType)) + 
    geom_bar(stat="identity") + 
    labs(subtitle = "Activation Difference", y = "Comp. - only MM") +
    coord_cartesian(ylim=c(-0.3, 0.3)) +
    scale_fill_manual(values = c(magma(7)[c(2,4,6)],"black")) +
    facet_grid(. ~ outcome, labeller = labeller(outcome = outLabs)) + 
    theme_minimal() + theme(legend.position = "none", axis.title.x = element_blank())
  
  temp$db$trialType <- factor(temp$db$trialType, levels = c("V+","A+","VA-","ctx"))
  levels(temp$db$trialType) <- c("V+","A*","VA-","ctx")
  outLabs <-  c("O1 (+)","O2 (*)"); names(outLabs) <- c("actO.1","actO.2")
  pAct <- ggplot(melt(temp$db,measure.vars = c("actO.1","actO.2")),
                 aes(x=nBlock,y=value,col=trialType,shape=trialType)) + 
    labs(x = "Blocks", y = "Activation", col="Trial Type",shape="Trial Type") +
    coord_cartesian(ylim=c(0, 1)) + scale_y_continuous(breaks = c(0.0,0.5,1.0)) +
    scale_shape_manual(values = c(19,19,21,3)) +
    scale_colour_manual(values = c(magma(7)[c(2,4,6)],"black")) +
    stat_summary(geom = "line") + stat_summary(fill="white") + theme_minimal() +
    facet_grid(. ~ variable, labeller = labeller(variable = outLabs))
  
  temp$wIH[temp$wIH==0] <- NA
  pWih <- ggplot(melt(temp$wIH),aes(x=Var1,y=Var2,fill=value)) + 
    labs(title = "L = 1",x = "Input", y = "Hidden", fill="Weight",
         subtitle = paste0("V vs A MM cor.: ",round(tempCor$cor,4))) +
    geom_tile() + theme_minimal() + 
    scale_x_continuous(breaks = 1:sum(param$nInput), labels = c("ctx","V","A")) +
    scale_y_continuous(breaks = 1:8, labels = paste0("h",1:8)) +
    scale_fill_gradient2(low="red",mid="white",high="green",
                         breaks=c(-3.5,0,3.5),labels=c(-3.5,0,3.5),
                         limits=c(-7,7))
  temp$wHO[temp$wHO==0] <- NA
  pWho <- ggplot(melt(temp$wHO),aes(x=Var1,y=Var2,fill=value)) + 
    labs(title = "L = 2", x = "Hidden", y = "Output", fill="Weight") +
    geom_tile() + theme_minimal() + coord_flip() +
    scale_x_continuous(breaks = 1:8, labels = paste0("h",1:8)) +
    scale_y_continuous(breaks = 1:2, labels = c("O1 (+)","O2 (*)")) +
    scale_fill_gradient2(low="red",mid="white",high="green",
                         breaks=c(-3.5,0,3.5),labels=c(-3.5,0,3.5),
                         limits=c(-7,7))
  plotsD[[i]] <- annotate_figure(ggarrange(ggarrange(pWih,pWho,ncol=2,align="h",common.legend=T,labels=c("A","B")),
                                           ggarrange(pAct,pActDiff,ncol=2,align="h",widths=c(6,4),labels=c("C","D")),
                                           nrow=2,heights=c(2,1)),
                                 top = text_grob("Differential Outcomes", face = "bold", size = 14))
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
}; remove(tempKO,tempCor,i,pWho,pWih,pAct,pActDiff)
dat2$actO <- ifelse(dat2$trialType == "V+",dat2$actO.1,
                    ifelse(dat2$trialType == "A+",dat2$actO.2,
                           (dat2$actO.1+dat2$actO.2)/2)) 
dat2$actO.1 <- dat2$actO.2 <- NULL
dat2$group <- "D"



training$OUTPUT <- t(matrix(c(1, # V+
                              1, # A+
                              0, # VA-
                              0, # VA-
                              0, # ctx / VA-
                              0, # ctx / VA-
                              0, # ctx / VA-
                              0, # ctx / VA-
                              0, # ctx / VA-
                              0),nrow=1,ncol=10))

plotsND <- list()
for (i in 1:nSim) {
  temp <- f_backProp(param,training,nKO.MM_one=0)
  temp$db$nBlock <- rep(1:(max(temp$db$nBlock)/10),each=100)
  if (i == 1) {
    dat1 <- data.frame(nSim = i,temp$db)
  } else {
    dat1 <- rbind(dat1,data.frame(nSim = i,temp$db))
  }
  
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  # algorithm to detect configural units
  for (j in 1:4) {
    wHO_KO <- temp$wHO; wHO_KO[c(1:2,7:8),] <- 0
    temp2 <- data.frame(group="D",nSim=i,
                        trialType=training$trialType[c(1:3,5)[j]],
                        actComp=param$actFun(param$actFun(training$INPUT[c(1:3,5)[j],] %*% temp$wIH) %*% temp$wHO),
                        actKO=param$actFun(param$actFun(training$INPUT[c(1:3,5)[j],] %*% temp$wIH) %*% wHO_KO))
    if (j == 1) {
      tempKO <- temp2
    } else {
      tempKO <- rbind(tempKO,temp2)
    }
  }; remove(temp2,j)
  tempKO$actCompMinKO <- tempKO$actComp - tempKO$actKO
  tempKO$trialType <- factor(tempKO$trialType, levels = c("V+","A+","VA-","ctx"))
  tempKO$outcome <- "O (+)"
  tempCor <- data.frame(group="D",nSim=i,cor=cor(temp$wIH[2,3:6],temp$wIH[3,3:6]))

  
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
  pActDiff <- ggplot(tempKO,aes(x=trialType,y=actCompMinKO,fill=trialType)) + 
    geom_bar(stat="identity") + 
    labs(subtitle = "Activation Difference", y = "Comp. - only MM") +
    coord_cartesian(ylim=c(-0.3, 0.3)) + 
    facet_grid(. ~ outcome) +
    scale_fill_manual(values = c(magma(7)[c(2,4,6)],"black")) + 
    theme_minimal() + theme(legend.position = "none", axis.title.x = element_blank())
  
  temp$db$trialType <- factor(temp$db$trialType, levels = c("V+","A+","VA-","ctx"))
  temp$db$outcome <- "O (+)"
  pAct <- ggplot(temp$db,aes(x=nBlock,y=actO,col=trialType,shape=trialType)) + 
    labs(x = "Blocks", y = "Activation", col="Trial Type",shape="Trial Type") +
    coord_cartesian(ylim=c(0, 1)) + scale_y_continuous(breaks = c(0.0,0.5,1.0)) +
    facet_grid(. ~ outcome) +
    scale_shape_manual(values = c(19,19,21,3)) +
    scale_colour_manual(values = c(magma(7)[c(2,4,6)],"black")) +
    stat_summary(geom = "line") + stat_summary(fill="white") + theme_minimal()
  
  temp$wIH[temp$wIH==0] <- NA
  pWih <- ggplot(melt(temp$wIH),aes(x=Var1,y=Var2,fill=value)) + 
    labs(title = "L = 1",x = "Input", y = "Hidden", fill="Weight",
         subtitle = paste0("V vs A MM cor.: ",round(tempCor$cor,4))) +
    geom_tile() + theme_minimal() + 
    scale_x_continuous(breaks = 1:sum(param$nInput), labels = c("ctx","V","A")) +
    scale_y_continuous(breaks = 1:8, labels = paste0("h",1:8)) +
    scale_fill_gradient2(low="red",mid="white",high="green",
                         breaks=c(-3.5,0,3.5),labels=c(-3.5,0,3.5),
                         limits=c(-7,7))
  temp$wHO[temp$wHO==0] <- NA
  pWho <- ggplot(melt(temp$wHO),aes(x=Var1,y=Var2,fill=value)) + 
    labs(title = "L = 2", x = "Hidden", y = "Output", fill="Weight") +
    geom_tile() + theme_minimal() + coord_flip() +
    scale_x_continuous(breaks = 1:8, labels = paste0("h",1:8)) +
    scale_y_continuous(breaks = 1, labels = "O (+)") +
    scale_fill_gradient2(low="red",mid="white",high="green",
                         breaks=c(-3.5,0,3.5),labels=c(-3.5,0,3.5),
                         limits=c(-7,7))
  plotsND[[i]] <- annotate_figure(ggarrange(ggarrange(pWih,pWho,ncol=2,align="h",common.legend=T,labels=c("A","B")),
                                            ggarrange(pAct,NULL,pActDiff,NULL,align="h",ncol=4,widths=c(5,1,3,1),
                                                      labels = c("C","","D","")),
                                            nrow=2,heights=c(2,1)),
                                 top = text_grob("Non-Differential Outcomes", face = "bold", size = 14))
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
}; remove(tempKO,tempCor,i,pWho,pWih,pAct,pActDiff)
dat1$group <- "ND"



# # # # # # # # # # # # print figures # # # # # # # # # # # # # # # # # # # # # 
print_fig <- 0
loc <- "C:/Users/scastiello/OneDrive - Nexus365/6.- lab people projects/Andy Delamater"
if (print_fig == 1) {
  for (i in 1:nSim) {
    message(paste("printing sim.:",i))
    ggsave(paste0(loc,"/weightFigures/Diff.Ind.",i,".png"),
           plot=plotsD[[i]],width=14,height=14,dpi=600,units="cm",limitsize=T)
    ggsave(paste0(loc,"/weightFigures/NonDiff.Ind.",i,".png"),
           plot=plotsND[[i]],width=14,height=14,dpi=600,units="cm",limitsize=T)
  }
}



# combine data bases
dat <- rbind(dat1,dat2)
# write.csv(dat,paste0(loc,"/veryGoodConfiguralSolutions.csv"))


datlf1 <- melt(dat, id.vars = c("nSim","nTrial","nBlock","trialType","group"))



# # # # Panel A # # # #
datlf1$trialType <- factor(datlf1$trialType,levels=c("V+","A+","VA-","ctx"))
labelLegend <- c("Flash (F+)","Noise (N+)","FN-","ctx")
datlf1$legend <- paste0(datlf1$trialType,datlf1$group)
datlf1$legend <- ifelse(datlf1$legend == "ctxD" | datlf1$legend == "ctxND", "ctx", datlf1$legend)
datlf1$legend <- factor(datlf1$legend, levels=c("V+D","V+ND","A+D","A+ND","VA-D","VA-ND","ctx"))

pA <- ggplot(datlf1,aes(x=nBlock,y=value,col=trialType,shape=legend)) + 
  geom_hline(yintercept = 0) +
  stat_summary(geom="line") + 
  stat_summary(fill="white") + 
  labs(y="Mean Output Activation",x="Blocks (100 trials)",col="Trial Type",shape="Trial Type") +
  scale_x_continuous(breaks = seq(0,20,5)) +
  scale_shape_manual(values = c(15,19,15,19,22,21,3)) +
  scale_colour_manual(labels = labelLegend, values = c(magma(7)[c(2,4,6)],"black")) +
  facet_grid(. ~ group,labeller=labeller(group = c("D"="Differential (D) Outcome",
                                                   "ND"="Non-Differential (ND) Outcome"))) +
  theme_bw() + theme(legend.position = c(0.64, 0.775),
                     legend.title = element_blank(),
                     axis.title.x = element_blank()) +
  guides(shape = F)
# pA



# # # # Panel B # # # #
datlf2 <- datlf1[datlf1$trialType != "ctx",]
datlf2$eleCom <- factor(ifelse(datlf2$trialType == "VA-","C","E"),levels = c("E","C"))

labelLegend <- c("Element (F+, N+)","Compound (FN-)")
datlf2$legend <- factor(paste0(datlf2$eleCom,datlf2$group),
                        levels=c("ED","END","CD","CND"))

pB <- ggplot(datlf2,aes(x=nBlock,y=value,col=eleCom,shape=legend)) + 
  geom_hline(yintercept = 0) +
  stat_summary(geom="line") + 
  stat_summary(fill="white") + 
  labs(y="Mean Output Activation",x="Blocks (100 trials)",shape="Trial Type") +
  scale_x_continuous(breaks = seq(0,20,5)) +
  scale_shape_manual(values = c(15,19,22,21)) +
  scale_colour_manual(labels = labelLegend, values = magma(7)[c(3,6)]) +
  facet_grid(. ~ group,labeller=labeller(group = c("D"="Differential (D) Outcome",
                                                   "ND"="Non-Differential (ND) Outcome"))) +
  theme_bw() + theme(legend.position = c(0.7, 0.8),
                     legend.title = element_blank(),
                     axis.title.x = element_blank()) +
  guides(shape = F)
# pB



# # # # Panel C # # # #
datlf3 <- as.data.frame(datlf2 %>% group_by(nSim,nBlock,group,eleCom) %>%
                          summarize(mValue = mean(value)))
temp <- datlf3[datlf3$eleCom == "C",]
datlf3 <- datlf3[datlf3$eleCom == "E",]
datlf3$mValue <- datlf3$mValue-temp$mValue
datlf3$eleCom <- NULL

labelLegend <- c("Differential (D) Outcome","Non-Differential (ND) Outcome")
pC <- ggplot(datlf3,aes(x=nBlock,y=mValue,shape=group)) + 
  geom_hline(yintercept = 0) +
  stat_summary(geom="line") + 
  stat_summary() + 
  labs(y="Element - Compound",x="Blocks (100 trials)",shape="Group") +
  scale_x_continuous(breaks = seq(0,20,5)) +
  scale_shape_manual(labels=labelLegend,values = c(15,19)) +
  theme_bw() + theme(axis.title.x = element_blank())
# pC
Cleg <- get_legend(pC)
pC <- pC + theme(legend.position = "none")

p <- annotate_figure(ggarrange(
  ggarrange(pA,pB,nrow=2, labels = c("A","B")),
  ggarrange(NULL,pC,Cleg,NULL,nrow=4,labels = c("","C","",""),heights = c(2,4,1,1)),
  widths = c(2,1.1)),
  top = text_grob("Negative Patterning Differential / Non-Differential Outcomes", face = "bold", size = 14),
  bottom = text_grob("Blocks (100 trials)", face = "bold", size = 14))
p



print_fig <- 0

loc <- "C:/Users/scastiello/OneDrive - Nexus365/6.- lab people projects/Andy Delamater"

if (print_fig == 1) {
  ggsave(paste0(loc,"/Figure2_8bl.100tr.tiff"),
         plot = p, width = 20, height = 18, dpi = 2400, units = "cm",limitsize = T)
}



######################### Figure 4 #############################################
############################################################################## #
############################################################################## #
# parameters
param <- list(a=0.3,b=0.9,nBlock=80,
              nHidden=data.frame(nHV=1,nHMM=64,nHA=1),
              nInput=data.frame(ctx=1,vis=1,aud=1),
              actFun=function (netin) {1/(1+exp(-(netin-2.2)))}
              )

training <- list(
  INPUT = t(matrix(c(1,  1,  0, # V+
                     1,  0,  1, # A+
                     1,  1,  1, # VA-
                     1,  1,  1, # VA-
                     1,  0,  0, # ctx
                     1,  0,  0, # ctx
                     1,  0,  0, # ctx
                     1,  0,  0, # ctx
                     1,  0,  0, # ctx
                     1,  0,  0),nrow=3,ncol=10)),
  # INPUT = t(matrix(c(1,  1,  0, # V+
  #                    1,  0,  1, # A+
  #                    1,  1,  1, # VA-
  #                    1,  1,  1, # VA-
  #                    1,  1,  1, # VA-
  #                    1,  1,  1, # VA-
  #                    1,  1,  1, # VA-
  #                    1,  1,  1, # VA-
  #                    1,  0,  0,
  #                    1,  0,  0),nrow=3,ncol=nTrTy)),
  #                  O1  O2
  OUTPUT = t(matrix(c(1, # V+
                      1, # A+
                      0, # VA-
                      0, # VA-
                      0, # VA-
                      0, # VA-
                      0, # VA-
                      0, # VA-
                      0,
                      0),nrow=1,ncol=nTrTy)),
  #               cxt   V   A
  TEST = t(matrix(c(1,  1,  0, # V+
                    1,  0,  1, # A+
                    1,  1,  1, # VA-
                    1,  1,  1, # VA-
                    1,  0,  0, # ctx
                    1,  0,  0, # ctx
                    1,  0,  0, # ctx
                    1,  0,  0, # ctx
                    1,  0,  0, # ctx
                    1,  0,  0),nrow=3,ncol=10)),
  # TEST = t(matrix(c(1,  1,  0, # V+
  #                   1,  0,  1, # A+
  #                   1,  1,  1, # VA-
  #                   1,  1,  1, # VA-
  #                   1,  1,  1, # VA-
  #                   1,  1,  1, # VA-
  #                   1,  1,  1, # VA-
  #                   1,  1,  1, # VA-
  #                   1,  0,  0,
  #                   1,  0,  0),nrow=3,ncol=nTrTy)),
  trialType = c("V+","A+","VA-","VA-","ctx","ctx","ctx","ctx","ctx","ctx")
  # trialType = c("V+","A+","VA-","VA-","VA-","VA-","VA-","VA-","ctx","ctx")
)

paramB <- paramA <- param


nSim <- 100

nKO.MM <- seq(0,64,4)#c(0,2,4,8,16,32,64,128) 
n.MM <- seq(0,64,4)

for (i in 1:length(nKO.MM)) {
  
  message(paste("MM:",n.MM[i]))
  
  # learning with different number of hidden unit 
  paramB$nHidden$nHMM <- n.MM[i]
  simsB <- f_run_sims(paramB,training,nSim,nKO.MM_one=0,KOdich=0)
  
  # KO 
  simsA <- f_run_sims(paramA,training,nSim,nKO.MM_one=nKO.MM[i],KOdich=1)
  
  if (i == 1) {
    actO <- simsB$actO_out
    actT <- simsB$actT_out
    actO.KO <- simsA$actO_out
    actT.KO <- simsA$actT_out
  } else {
    actO <- rbind(actO,simsB$actO_out)
    actT <- rbind(actT,simsB$actT_out)
    actO.KO <- rbind(actO.KO,simsA$actO_out)
    actT.KO <- rbind(actT.KO,simsA$actT_out)
  }
}



# plot_breaks <- seq(min(n.MM),max(n.MM),by=round(length(n.MM)/2))
plot_breaks <- round(seq(min(n.MM),max(n.MM),by=(max(n.MM)-min(n.MM))/5))
# plot_labels <- round((seq(min(n.MM),max(n.MM),by=round(length(n.MM)/2))/max(n.MM))*100,2)
plot_labels <- round((seq(min(n.MM),max(n.MM),by=(max(n.MM)-min(n.MM))/5)/max(n.MM))*100,2)

subTitLab <- paste0("alpha=",param$a,"; beta=",param$b,"; #MM=",param$nHidden$nHMM)

#### Trial Types. Prepare factor trial types ####
# KO before
actT$trialType <- factor(actT$trialType,levels=c("V+","A+","VA-","ctx"))
levels(actT$trialType) <- c("F+","N+","FN-","ctx")
actT$nHMM <- as.factor(actT$nHMM)
levels(actT$nHMM) <- rev(levels(actT$nHMM))
actT$nHMM <- as.integer(as.character(actT$nHMM))
# KO after
actT.KO$trialType <- factor(actT.KO$trialType,levels=c("V+","A+","VA-","ctx"))
levels(actT.KO$trialType) <- c("F+","N+","FN-","ctx")


library(ggplot2); library(viridis)
pA <- ggplot(actT, aes(x=nHMM,y=actTest,col=trialType,shape=trialType)) + 
  labs(x="% of KO MM hidden units",
       y="Output Activation", col="Trial Type", shape="Trial Type") +
  stat_summary(geom="line",position = position_dodge(0.5)) +
  stat_summary(position = position_dodge(0.5), fill="white") +
  scale_y_continuous(breaks = seq(0,1,by=0.5)) +
  scale_x_continuous(breaks = plot_breaks,labels = plot_labels) +
  scale_colour_manual(values = c(magma(7)[c(2,4,6)],"black")) +
  scale_shape_manual(values = c(19,19,21,3)) +
  coord_cartesian(ylim=c(0, 1)) +
  theme_bw() + guides(col=guide_legend(nrow=2,byrow=TRUE),
                      shape=guide_legend(nrow=2,byrow=TRUE)) +
  theme(legend.position = c(0.5, 0.5),
        # legend.direction = "horizontal",
        legend.title = element_blank(),
        axis.title = element_blank())
# pA


pB <- ggplot(actT.KO, aes(x=nHMM,y=actTest,col=trialType,shape=trialType)) + 
  labs(x="% of KO MM hidden units",
       y="Output Activation", col="Trial Type", shape="Trial Type") +
  stat_summary(geom="line",position = position_dodge(0.5)) +
  stat_summary(position = position_dodge(0.5), fill="white") +
  scale_y_continuous(breaks = seq(0,1,by=0.5)) +
  scale_x_continuous(breaks = plot_breaks,labels = plot_labels) +
  scale_colour_manual(values = c(magma(7)[c(2,4,6)],"black")) +
  scale_shape_manual(values = c(15,15,22,3)) +
  coord_cartesian(ylim=c(0, 1)) +
  theme_bw() + guides(col=guide_legend(nrow=2,byrow=TRUE),
                      shape=guide_legend(nrow=2,byrow=TRUE)) + 
  theme(legend.position = c(0.7, 0.8),
        # legend.direction = "horizontal",
        legend.title = element_blank(),
        axis.title = element_blank())
# pB



#### Element vs Compound. Prepare factor trial types ####
# KO before
actT$trialType <- as.character(actT$trialType)
temp1 <- actT[actT$trialType == "FN-",]
temp1$trialType <- "Compound (FN-)"
temp2 <- actT[actT$trialType == "F+",]
temp2$trialType <- "Element (F+, N+)"
temp2$actTest <- (temp2$actTest+actT$actTest[actT$trialType == "N+"])/2
actT_eleComp <- rbind(temp2,temp1)
actT_eleComp$trialType <- factor(actT_eleComp$trialType,
                                 levels=unique(actT_eleComp$trialType))
# KO after
actT.KO$trialType <- as.character(actT.KO$trialType)
temp1 <- actT.KO[actT.KO$trialType == "FN-",]
temp1$trialType <- "Compound (FN-)"
temp2 <- actT.KO[actT.KO$trialType == "F+",]
temp2$trialType <- "Element (F+, N+)"
temp2$actTest <- (temp2$actTest+actT.KO$actTest[actT.KO$trialType == "N+"])/2
actT.KO_eleComp <- rbind(temp2,temp1)
actT.KO_eleComp$trialType <- factor(actT.KO_eleComp$trialType,
                                    levels=unique(actT.KO_eleComp$trialType))


pC <- ggplot(actT_eleComp, aes(x=nHMM,y=actTest,col=trialType,shape=trialType)) + 
  labs(x="% of KO MM hidden units",
       y="Output Activation", col="Trial Type", shape="Trial Type") +
  stat_summary(geom="line",position = position_dodge(0.5)) +
  stat_summary(position = position_dodge(0.5), fill="white") +
  scale_y_continuous(breaks = seq(0,1,by=0.5)) +
  scale_x_continuous(breaks = plot_breaks,labels = plot_labels) +
  scale_shape_manual(values = c(19,21)) +
  scale_colour_manual(values = magma(7)[c(3,6)]) +
  coord_cartesian(ylim=c(0, 1)) +
  theme_bw() + guides(col=guide_legend(nrow=2,byrow=TRUE),
                      shape=guide_legend(nrow=2,byrow=TRUE)) +
  theme(legend.position = c(0.4, 0.5),
        # legend.direction = "horizontal",
        legend.title = element_blank(),
        axis.title = element_blank())
# pC

pD <- ggplot(actT.KO_eleComp, aes(x=nHMM,y=actTest,col=trialType,shape=trialType)) + 
  labs(x="% of KO MM hidden units",
       y="Output Activation", col="Trial Type", shape="Trial Type") +
  stat_summary(geom="line",position = position_dodge(0.5)) +
  stat_summary(position = position_dodge(0.5), fill="white") +
  scale_y_continuous(breaks = seq(0,1,by=0.5)) +
  scale_x_continuous(breaks = plot_breaks,labels = plot_labels) +
  scale_shape_manual(values = c(15,22)) +
  scale_colour_manual(values = magma(7)[c(3,6)]) +
  coord_cartesian(ylim=c(0, 1)) +
  theme_bw() + guides(col=guide_legend(nrow=2,byrow=TRUE),
                      shape=guide_legend(nrow=2,byrow=TRUE)) +
  theme(legend.position = c(0.7, 0.8),
        # legend.direction = "horizontal",
        legend.title = element_blank(),
        axis.title = element_blank())
# pD

#### Element vs Compound. Prepare factor trial types ####
library(dplyr)
# KO before
tempB <- actT_eleComp %>% group_by(nHMM,nSim,trialType) %>% summarise(actTest = mean(actTest))
temp1 <- tempB[tempB$trialType == "Element (F+, N+)",]
temp1$trialType <- "KO before"
temp1$actTest <- temp1$actTest - tempB$actTest[tempB$trialType == "Compound (FN-)"]
# KO after
tempA <- actT.KO_eleComp %>% group_by(nHMM,nSim,trialType) %>% summarise(actTest = mean(actTest))
temp2 <- tempA[tempA$trialType == "Element (F+, N+)",]
temp2$trialType <- "KO after"
temp2$actTest <- temp2$actTest - tempA$actTest[tempA$trialType == "Compound (FN-)"]

actT_diff <- rbind(temp2,temp1)

pE <- ggplot(actT_diff, aes(x=nHMM,y=actTest,shape=trialType)) + 
  labs(x="% of KO MM hidden units",
       y="Element - Compound", shape="Trial Type") +
  stat_summary(geom="line",position = position_dodge(0.5)) +
  stat_summary(position = position_dodge(0.5), fill="white") +
  scale_y_continuous(breaks = seq(0,1,by=0.5)) +
  scale_x_continuous(breaks = plot_breaks,labels = plot_labels) +
  scale_shape_manual(values = c(15,21)) +
  coord_cartesian(ylim=c(0, 1)) +
  theme_bw() + guides(col=guide_legend(nrow=2,byrow=TRUE),
                      shape=guide_legend(nrow=2,byrow=TRUE)) +
  theme(legend.position = c(0.209, 0.25),
        # legend.direction = "horizontal",
        # plot.margin = margin(0, 0, 0, 0.5, "cm"),
        legend.title = element_blank(),
        axis.title = element_blank())
# pE


library(ggpubr)
pFinal <- annotate_figure(ggarrange(annotate_figure(
  ggarrange(annotate_figure(ggarrange(pA,pC,nrow=2,ncol=1,align = "hv",labels = c("A","C"),
                                      label.x = 0.05,label.y = 1.08),
                            top=text_grob("Before Training",face="bold",size=16)),
                            # bottom=text_grob("% of KO MM hidden units",face="bold",size=14)),
            annotate_figure(ggarrange(pB,pD,nrow=2,ncol=1,align = "hv",labels = c("B","D"),
                                      label.x = 0.05,label.y = 1.08),
                            top=text_grob("After Training",face="bold",size=16)),
                            # bottom=text_grob("% of KO MM hidden units",face="bold",size=14)),
            ncol=2),
  left=text_grob("Output Activation",face="bold",size=14,rot=90)),
  ggarrange(NULL,
            annotate_figure(ggarrange(pE,labels=c("E"),label.x = 0.05,label.y = 1.08),
                            top=text_grob("After vs. Before",face="bold",size=16),
                            left=text_grob("Element - Compound",face="bold",size=14,rot=90)),
            NULL,ncol=3,widths=c(1,2,1)),
  nrow=2,heights=c(2,1)),
  bottom=text_grob("% of Multi-Modal Units Knocked Out",face="bold",size=14))
pFinal



print_fig <- 0

loc <- "C:/Users/scastiello/OneDrive - Nexus365/6.- lab people projects/Andy Delamater"

if (print_fig == 1) {
  ggsave(paste0(loc,"/Figure4_8bl.100tr.tiff"),
         plot = pFinal,width = 16,height = 20,dpi = 2400,units = "cm",limitsize = T)
}




