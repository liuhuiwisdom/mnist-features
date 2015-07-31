library(deepnet)

mnist.data <- load.mnist(dir = "./data/")
n.train <- mnist.data$train$n
x.train <- mnist.data$train$x
y.train <- mnist.data$train$y

n.test <- mnist.data$test$n
x.test <- mnist.data$test$x
y.test <- mnist.data$test$y

# Train a Deep neural network with weights initialized by Stacked AutoEncoder:
dnn <- sae.dnn.train(x.train, y.train, hidden=c(1000,1000,1000), sae_output="linear", numepochs=5, batchsize = 100, hidden_dropout = 0.1)
