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
    conIH[(param$nInput$ctx+1):(param$nInput$ctx+param$nInput$vis),
          (nHid-param$nHidden$nHA+1):nHid] <- 0}
  # block auditory input with visual hidden
  if (param$nHidden$nHV > 0) {
    conIH[(param$nInput$ctx+param$nInput$vis+1):nStim,
          1:param$nHidden$nHV] <- 0}
  wIH <- wIH * conIH
  
  dwHO <- matrix(0, nrow = nHid, ncol = nOut)
  wHO <- matrix(runif(nHid*nOut)-0.5, nrow = nHid) # rows hidden units, cols output units or options
  
  
  #### backprop algorithm ###
  # see also: https://web.stanford.edu/group/pdplab/pdphandbook/handbookch6.html
  for (nB in 1:nBlock) {
    
    for (t in 1:nTrialType) {
      
      ### ### activation ### ### forward activation ## chain rule: act_fun(act_fun(trialBlock %*% wIH) %*% wHO)
      trial <- t+((nB-1)*nTrialType)
      
      actH[trial,] <- actFun(INPUT[t,] %*% wIH) # INPUT[t,] assumed to be a 1 x nStim matrix
      
      actO[trial,] <- actFun(actH[trial,] %*% wHO) # actH[t,] assumed to be a 1 x nOut matrix
      
      
      
      ### ### learning (weight changes) ### ###
      
      ### delta (error) for output
      deltaO[trial,] <- (OUTPUT[t,] - actO[trial,]) * (actO[trial,] * (1 - actO[trial,]))
      
      # weight change Hidden-Output
      dwHO[,] <- t(a*(as.matrix(deltaO[trial,]) %*% actH[trial,])) + b*dwHO[,]
      
      # update weights Hidden-Output
      wHO[,] <- wHO[,] + dwHO[,]
      
      
      ### delta (error) for hidden
      deltaH[trial,] <- (actH[trial,]*(1-actH[trial,])) * (deltaO[trial,]%*%t(wHO[,])) #rowSums(as.matrix(deltaO[t,] * wHO[,]))
      
      # weight change Input-Hidden
      dwIH[,] <- t(a*(as.matrix(deltaH[trial,]) %*% INPUT[t,])) + b*dwIH[,]
      
      
      # update weights Input-Hidden
      wIH[,] <- wIH[,] + (dwIH[,] * conIH)
      
    } # end trials per block cycle # t
    
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
                   trialType=rep(training$trialType,nBlock),
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



######################### Figure 4 #############################################
############################################################################## #
############################################################################## #
# parameters
param <- list(a=0.3,b=0.9,nBlock=100,
              nHidden=data.frame(nHV=1,nHMM=64,nHA=1),
              nInput=data.frame(ctx=1,vis=2,aud=2),
              actFun=function (netin) {1/(1+exp(-(netin-2.2)))}
              )

#                                 cxt   V share V
training <- list(
                 # INPUT = t(matrix(c(1,  1,  1,  0, # V+
                 #                    1,  0,  1,  1, # A+
                 #                    1,  1,  1,  1, # VA-
                 #                    1,  0,  0,  0),nrow=4,ncol=4)),
#                                   ctx     V     A             
                 INPUT = t(matrix(c(1,  1,1,  0,0, # V+
                                    1,  0,0,  1,1, # A+
                                    1,  1,1,  1,1, # VA-
                                    1,  1,1,  1,1, # VA-
                                    1,  1,1,  1,1, # VA-
                                    1,  1,1,  1,1, # VA-
                                    1,  1,1,  1,1, # VA-
                                    1,  1,1,  1,1, # VA-
                                    1,  0,0,  0,0),nrow=5,ncol=9)),
                 #                  O1  O2
                 # OUTPUT = t(matrix(c(1,  0, # V+
                 #                     0,  1, # A+
                 #                     0,  0, # VA-
                 #                     0,  0),nrow=2,ncol=4)),
                 OUTPUT = t(matrix(c(1, # V+
                                     1, # A+
                                     0, # VA-
                                     0, # VA-
                                     0, # VA-
                                     0, # VA-
                                     0, # VA-
                                     0, # VA-
                                     0),nrow=1,ncol=9)),
                 #               cxt     V     A
                 TEST = t(matrix(c(1,  1,1,  0,0, # V+
                                   1,  0,0,  1,1, # A+
                                   1,  1,1,  1,1, # VA-
                                   1,  1,1,  1,1, # VA-
                                   1,  1,1,  1,1, # VA-
                                   1,  1,1,  1,1, # VA-
                                   1,  1,1,  1,1, # VA-
                                   1,  1,1,  1,1, # VA-
                                   1,  0,0,  0,0),nrow=5,ncol=9)),
                  #               cxt   V share V
                  # TEST = t(matrix(c(1,  1,  1,  0, # V+
                  #                   1,  0,  1,  1, # A+
                  #                   1,  1,  1,  1, # VA-
                  #                   1,  0,  0,  0),nrow=4,ncol=4)),

                 trialType = c("V+","A+","VA-","VA-","VA-","VA-","VA-","VA-","ctx")
                 )
paramB <- paramA <- param


nSim <- 100

nKO.MM <- seq(0,64,16)#c(0,2,4,8,16,32,64,128) 
n.MM <- seq(0,64,16)

library(reshape2)
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

  

library(ggplot2)
# actO$trialType <- factor(actO$trialType,levels=c("A+","V+","VA-","ctx")) 
# p <- ggplot(actO[actO$nHMM == 0,], aes(x=nBlock,y=value,col=trialType)) + 
#   labs(title = "alpha=0.3; beta=0.9; nHid=2 (V:0, MM:2, A:0)") +
#   stat_summary() +
#   scale_y_continuous(breaks = seq(0,1,by=0.25)) +
#   coord_cartesian(ylim=c(0, 1)) +
#   facet_grid(.~variable) +
#   theme_bw()
# p
# p <- ggplot(actO.KO[actO.KO$nHMM == 0,], aes(x=nBlock,y=value,col=trialType)) + 
#   labs(title = "alpha=0.3; beta=0.9; nHid=2 (V:0, MM:2, A:0)") +
#   stat_summary() +
#   scale_y_continuous(breaks = seq(0,1,by=0.25)) +
#   coord_cartesian(ylim=c(0, 1)) +
#   facet_grid(.~variable) +
#   theme_bw()
# p


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


library(viridis)
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
  scale_shape_manual(values = c(15,19)) +
  coord_cartesian(ylim=c(0, 1)) +
  theme_bw() + guides(col=guide_legend(nrow=2,byrow=TRUE),
                      shape=guide_legend(nrow=2,byrow=TRUE)) +
  theme(legend.position = c(0.25, 0.25),
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
  bottom=text_grob("% of Knockout Multi-Modal Units",face="bold",size=14))
pFinal

# p <- ggarrange(pA,pB,pC,pD,pE,NULL,labels=c(LETTERS[1:5],""),nrow=3,ncol=2)
# p


print_fig <- 0

loc <- "C:/Users/scastiello/OneDrive - Nexus365/6.- lab people projects/Andy Delamater"

if (print_fig == 1) {
  ggsave(paste0(loc,"/p064_v1.png"),
         plot = pFinal,width = 16,height = 20,dpi = 2400,units = "cm",limitsize = T)
}



######################### Figure 2 #############################################
############################################################################## #
############################################################################## #
# parameters
param <- list(a=0.3,b=0.9,nBlock=100,
              nHidden=data.frame(nHV=2,nHMM=4,nHA=2),
              nInput=data.frame(ctx=1,vis=2,aud=2),
              actFun=function (netin) {1/(1+exp(-(netin-2.2)))}
)

training <- list(
  INPUT = t(matrix(c(1,  1,1,  0,0, # V+
                     1,  0,0,  1,1, # A+
                     1,  1,1,  1,1, # VA-
                     1,  1,1,  1,1, # VA-
                     1,  0,0,  0,0, # ctx 
                     1,  0,0,  0,0, # ctx
                     1,  0,0,  0,0, # ctx
                     1,  0,0,  0,0, # ctx
                     1,  0,0,  0,0, # ctx
                     1,  0,0,  0,0),nrow=5,ncol=10)),
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
  #               cxt     V     A
  TEST = t(matrix(c(1,  1,1,  0,0, # V+
                    1,  0,0,  1,1, # A+
                    1,  1,1,  1,1, # VA-
                    1,  1,1,  1,1, # VA-
                    1,  0,0,  0,0, # ctx
                    1,  0,0,  0,0, # ctx
                    1,  0,0,  0,0, # ctx
                    1,  0,0,  0,0, # ctx
                    1,  0,0,  0,0, # ctx
                    1,  0,0,  0,0),nrow=5,ncol=10)),
  trialType = c("V+","A+","VA-","VA-","ctx","ctx","ctx","ctx","ctx","ctx")
)

nSim <- 32
for (i in 1:nSim) {
  temp <- f_backProp(param,training,nKO.MM_one=0)
  temp$db$nBlock <- rep(1:20,each=50)
  if (i == 1) {
    dat2 <- data.frame(nSim = i,temp$db)
  } else {
    dat2 <- rbind(dat2,data.frame(nSim = i,temp$db))
  }
}
dat2$actO <- ifelse(dat2$trialType == "V+",dat2$actO.1,
                   ifelse(dat2$trialType == "A+",dat2$actO.2,
                          (dat2$actO.1+dat2$actO.2)/2)) 
dat2$actO.1 <- dat2$actO.2 <- NULL
dat2$group <- "D"

training$OUTPUT <- t(matrix(c(1, # V+
                              1, # A+
                              0, # VA-
                              0, # VA-
                              0, # ctx
                              0, # ctx
                              0, # ctx
                              0, # ctx
                              0, # ctx
                              0),nrow=1,ncol=10))

for (i in 1:nSim) {
  temp <- f_backProp(param,training,nKO.MM_one=0)
  temp$db$nBlock <- rep(1:20,each=50)
  if (i == 1) {
    dat1 <- data.frame(nSim = i,temp$db)
  } else {
    dat1 <- rbind(dat1,data.frame(nSim = i,temp$db))
  }
}
dat1$group <- "ND"



dat <- rbind(dat1,dat2)

library(reshape2)
library(ggplot2)
library(viridis)

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
  labs(y="Mean Output Activation",x="Blocks (5 trials)",col="Trial Type",shape="Trial Type") +
  scale_x_continuous(breaks = seq(0,20,5)) +
  scale_shape_manual(values = c(15,19,15,19,22,21,3)) +
  scale_colour_manual(labels = labelLegend, values = c(magma(7)[c(2,4,6)],"black")) +
  facet_grid(. ~ group,labeller=labeller(group = c("D"="Differential (D) Outcome",
                                                   "ND"="Non-Differential (ND) Outcome"))) +
  theme_bw() + theme(legend.position = c(0.64, 0.775),
                     legend.title = element_blank(),
                     axis.title.x = element_blank()) +
  guides(shape = F)
pA



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
  labs(y="Mean Output Activation",x="Blocks (5 trials)",shape="Trial Type") +
  scale_x_continuous(breaks = seq(0,20,5)) +
  scale_shape_manual(values = c(15,19,22,21)) +
  scale_colour_manual(labels = labelLegend, values = magma(7)[c(3,6)]) +
  facet_grid(. ~ group,labeller=labeller(group = c("D"="Differential (D) Outcome",
                                                   "ND"="Non-Differential (ND) Outcome"))) +
  theme_bw() + theme(legend.position = c(0.7, 0.8),
                     legend.title = element_blank(),
                     axis.title.x = element_blank()) +
  guides(shape = F)
pB



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
  labs(y="Element - Compound",x="Blocks (5 trials)",shape="Group") +
  scale_x_continuous(breaks = seq(0,20,5)) +
  scale_shape_manual(labels=labelLegend,values = c(15,19)) +
  theme_bw() + theme(axis.title.x = element_blank())
pC
Cleg <- get_legend(pC)
pC <- pC + theme(legend.position = "none")

library(ggpubr)
p <- annotate_figure(ggarrange(
  ggarrange(pA,pB,nrow=2, labels = c("A","B")),
  ggarrange(NULL,pC,Cleg,NULL,nrow=4,labels = c("","C","",""),heights = c(2,4,1,1)),
  widths = c(2,1.1)),
  top = text_grob("Negative Patterning Differential / Non-Differential Outcomes", face = "bold", size = 14),
  bottom = text_grob("Blocks (5 trials)", face = "bold", size = 14))
p

print_fig <- 0

loc <- "C:/Users/scastiello/OneDrive - Nexus365/6.- lab people projects/Andy Delamater"

if (print_figure == 1) {
  ggsave(paste0(loc,"/Figure2_v1.png"),
         plot = p, width = 20, height = 18, dpi = 2400, units = "cm",limitsize = T)
}
