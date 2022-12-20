library(mvtnorm)
library(plotly)
library(Rcpp)
library(RcppArmadillo)
library(salso)
library(DPSPN)

mu1 <- c(0, 0, 4)
sig1 <- matrix(c(1, 0, -0.8,
                 0, 1, 0,
                 -0.8, 0, 1), 3, 3)
mu2 <- c(2, 0, -2)
sig2 <- matrix(c(1, 0, 0,
                 0, 1, -0.8,
                 0, -0.8, 1), 3, 3)
mu3 <- c(-2, 0.2, 0)
sig3 <- matrix(c(1, 0, 0,
                 0, 1, 0,
                 0, 0, 1), 3, 3)
N <- 500

datDP <- rbind(rmvnorm(N, mu1, sig1),
               rmvnorm(N, mu2, sig2), 
               rmvnorm(N, mu3, sig3))

trueLabel <- c(rep(1,N), rep(2,N), rep(3,N))
model <- fitDPSPN(datDP, 1, 2, nSamp = 100, nBurn = 500, nThin = 2)
estLabel <-  as.vector(salso(t(model$z), maxZealousAttempts = 0))

circular <- atan2(datDP[,2], datDP[,1])
linear <- datDP[,3]

plot_ly(x = ~cos(circular), y = ~sin(circular), z = ~linear,
        size = 5, split = ~trueLabel, scene = 'scene1',
        type="scatter3d", mode = "markers") %>%
  layout(title = 'Ground Truth')
plot_ly(x = ~cos(circular), y = ~sin(circular), z = ~linear,
        size = 5, split = ~estLabel,  scene = 'scene2',
        type="scatter3d", mode = "markers") %>%
  layout(title = 'DPSPN Clustering')



datHDP <- rbind(rmvnorm(N, mu1, sig1),
                rmvnorm(N, mu2, sig2), 
                rmvnorm(N, mu1, sig1),
                rmvnorm(N, mu3, sig3),
                rmvnorm(N, mu2, sig2),
                rmvnorm(N, mu3, sig3))
rid <- c(rep(1,2*N), rep(2,2*N), rep(3,2*N))
model <- fitHDPSPN(datHDP, rid, 1, 2, nSamp = 100, nBurn = 500, nThin = 2)
trueLabel <- c(rep(1,N), rep(2,N),
               rep(1,N), rep(3,N),
               rep(2,N), rep(3,N))
estLabel <-  as.vector(salso(t(model$z), maxZealousAttempts = 0))
circular <- atan2(datHDP[,2], datHDP[,1])
linear <- datHDP[,3]
plot_ly(x = ~cos(circular), y = ~sin(circular), z = ~linear,
        size = 5, split = ~trueLabel, scene = 'scene1',
        type="scatter3d", mode = "markers") %>%
  layout(title = 'Ground Truth')
plot_ly(x = ~cos(circular), y = ~sin(circular), z = ~linear,
        size = 5, split = ~estLabel,  scene = 'scene2',
        type="scatter3d", mode = "markers") %>%
  layout(title = 'DPSPN Clustering')

res <- computeLoglikhd(datDP, 2, model$param)

