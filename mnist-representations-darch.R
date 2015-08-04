library(deepnet)
library(darch)

mnist.data <- load.mnist(dir = "./data/")
n.train <- mnist.data$train$n
x.train <- mnist.data$train$x
y.train <- mnist.data$train$y

n.test <- mnist.data$test$n
x.test <- mnist.data$test$x
y.test <- mnist.data$test$y

# Generating the darch
darch <- newDArch(c(28*28,1000,10),batchSize=100)

# Pre-Train the darch
darch <- preTrainDArch(darch,x.train,maxEpoch=10)

# Prepare the layers for backpropagation training (fine-tuning)
# The layer functions must be
# set to the unit functions which also calculate derivatives of the function result (sigmoidUnitDerivative).
layers <- getLayers(darch)
for(i in length(layers):1){
  layers[[i]][[2]] <- sigmoidUnitDerivative
}
setLayers(darch) <- layers
rm(layers)

# Setting and running the Fine-Tune function
setFineTuneFunction(darch) <- backpropagation
darch <- fineTuneDArch(darch,x.train,as.matrix(y.train),maxEpoch=1000)


# Running the darch
darch <- darch <- getExecuteFunction(darch)(darch,x.train)
outputs <- getExecOutputs(darch)
cat(outputs[[length(outputs)]])