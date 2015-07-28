# Test my stacked autoencoder trained by SAENET.train on the feature extraction from MNIST images, to see whether it finds interesting features of MNIST characters.
rm(list=ls())
####### Parameters of the run: ######################################################################################################################################
seed <- 12345L
classifier.type = "rf"         # type of classifier used. "rf" for Random Forest, "svm-rbf" for SVM with Radial Basis Function kernel. 
performance.measure = "accuracy"  #model performace measure
####### END of Parameters of the run ######################################################################################################################################

### FUNCTIONS: ###
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Load the MNIST digit recognition dataset into R
# http://yann.lecun.com/exdb/mnist/
# assume you have all 4 files and gunzip'd them
# creates train$n, train$x, train$y  and test$n, test$x, test$y
# e.g. train$x is a 60000 x 784 matrix, each row is one digit (28x28)
# call:  show_digit(train$x[5,])   to see a digit.
# brendan o'connor - gist.github.com/39760 - anyall.org

load_mnist <- function(folder) {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    ret$nrow = nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ret$ncol = ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file(paste(folder,'train-images-idx3-ubyte',sep=""))
  test <<- load_image_file(paste(folder,'t10k-images-idx3-ubyte',sep=""))
  
  train$y <<- load_label_file(paste(folder,'train-labels-idx1-ubyte',sep=""))
  test$y <<- load_label_file(paste(folder,'t10k-labels-idx1-ubyte',sep=""))  
}
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
show_digit <- function(arr784, nrow=28, ncol=28, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=nrow)[,ncol:1], col=col, ...)
}
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
train_model <- function(examples, hp, model.type=NULL) #train a classifier of the given model.type on the given examples (assuming the 1st column is target "attribute.type"), with given hyperparameter vector hp
{  
  require(randomForest)
  if (model.type=="rf") {  #Random Forest
    model <- randomForest(as.factor(target) ~ ., data=examples, importance=TRUE)
  } else {
    stop(paste("The requested model type \'",model.type,"\' is not implemented yet. Stopping."))
  }
  return(list("model"=model))
}
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
train_2classifiers.per.hp <- function(instances.train, hp, verbose=FALSE, worker.id="")  #Train 2 classifiers (classifier on raw features and classifier on autoencoder) on examples from instances.train, for given hp vector
{
  require(autoencoder)
  require(SAENET)
  examples.train <- train$x[instances.train,]   #extract training records (image vectors) corresponding to instances.train
  targets.train <- train$y[instances.train]   #extract targets (image classes) corresponding to instances.train
  examples.train <- cbind(target=targets.train,as.data.frame(examples.train))
  # if (verbose) visualize_examples(examples.train, "raw representation of training examples")
  classifier_on_raw <- train_model(examples=examples.train, hp, model.type=classifier.type)$model
  if (is.null(classifier_on_raw)) stop("classifier_on_raw is NULL. Stopping.")
  
  ## Train the autoencoder to discover condensed features:
  n.hidden.units <- NULL
  for (l in 1:hp$hidden.layers) n.hidden.units <- c(n.hidden.units,hp[,paste("n.units.",l,sep="")])
  if (verbose) cat(worker.id,"Training an autoencoder with",hp$hidden.layers,"hidden layers, n.hidden.units =",n.hidden.units,", lambda =",hp$lambda,", beta =",hp$beta,", rho =",hp$rho,"...\n")
  encoder <- SAENET.train(as.matrix(examples.train[,-1]), n.nodes = n.hidden.units,
                          lambda = hp$lambda, beta = hp$beta, rho = hp$rho, epsilon = 1./sqrt(max(n.hidden.units)), rescale.flag=T, optim.method="CG", 
                          max.iterations = 10000)
  
  ### Using the trained autoencoder, calculate the compressed feature vectors of the training example set examples.train:
  compressed_examples.train <- SAENET.predict(encoder, new.data=as.matrix(examples.train[,-1]), layers = length(encoder))
  compressed_examples.train <- cbind(examples.train$target, as.data.frame(compressed_examples.train[[1]]$X.output))
  colnames(compressed_examples.train)[1] <- c("target")
  # if (verbose) visualize_examples(compressed_examples.train, "compressed representation of training examples")
  ### Now, train a classifier on compressed_examples.train:
  classifier_on_autoencoder <- train_model(examples=compressed_examples.train, hp=hp, model.type=classifier.type)$model
  if (is.null(classifier_on_autoencoder)) stop("classifier_on_autoencoder is NULL. Stopping.")
  
  return(list("hp"=hp,"classifier_on_raw"=classifier_on_raw,"classifier_on_autoencoder"=classifier_on_autoencoder,"encoder"=encoder))   #return the trained classifiers
}
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
evaluate_2classifiers.per.hp <- function(classifiers, instances.validate, hp, measure, verbose=FALSE)  #Evaluate 2 classifiers (classifier on raw features and classifier on autoencoder) on examples from instances.validate, for given hp vector
{
  require(caret)
  
  if(verbose) cat("Evaluating classifiers...")
  examples.validate <- train$x[instances.validate,]   #extract validating records (image vectors) corresponding to instances.validate
  targets.validate <- train$y[instances.validate]     #extract targets (image classes) corresponding to instances.validate
  examples.validate <- cbind(target=targets.validate,as.data.frame(examples.validate))
  # if (verbose) visualize_examples(examples.validate, "raw representation of validation examples")
  classifier_on_raw <- classifiers$classifier_on_raw
  
  ## Extract the autoencoder from classifiers:
  encoder <- classifiers$encoder
  
  ### Using the autoencoder, calculate the compressed feature vectors of the validation example set examples.validate:
  compressed_examples.validate <- SAENET.predict(encoder, new.data=as.matrix(examples.validate[,-1]), layers = length(encoder))
  compressed_examples.validate <- cbind(examples.validate$target, as.data.frame(compressed_examples.validate[[1]]$X.output))
  colnames(compressed_examples.validate)[1] <- c("target")
  # if (verbose) visualize_examples(compressed_examples.validate, "compressed representation of validating examples")
  ### Now, extract a classifier on autoencoder: 
  classifier_on_autoencoder <- classifiers$classifier_on_autoencoder
  
  ### Calculate the performance (based on measure) of classifier_on_raw and classifier_on_autoencoder on examples.validate and compressed_examples.validate:
  # Predict attribute types with each classifier:
  examples.validate$predicted.target <- predict(classifier_on_raw,newdata=examples.validate[,!colnames(examples.validate) %in% "target"], type="response")
  compressed_examples.validate$predicted.target <- predict(classifier_on_autoencoder,newdata=compressed_examples.validate[,!colnames(compressed_examples.validate) %in% "target"], type="response")
  # Build confusion matrix for each classifier:
  confusionMatrix_classifier_on_raw <- with(examples.validate, caret::confusionMatrix(data=predicted.target, reference=target))
  confusionMatrix_classifier_on_autoencoder <- with(compressed_examples.validate, caret::confusionMatrix(data=predicted.target, reference=target))
  # Extract the performance measure in accordance with measure:
  if (measure=="accuracy"){
    perf.classifier_on_raw <- confusionMatrix_classifier_on_raw$overall[1]
    perf.classifier_on_autoencoder <- confusionMatrix_classifier_on_autoencoder$overall[1]
    perf.ratio <- perf.classifier_on_autoencoder/perf.classifier_on_raw
  } else stop(paste("The requested model performance measure",measure,"is not yet implemented. Stopping."))
  
  if(verbose){
    cat("done\n")
    cat(paste("Evaluated performances: perf.classifier_on_raw=",perf.classifier_on_raw,", perf.classifier_on_autoencoder=",perf.classifier_on_autoencoder,", perf.ratio=",perf.ratio,"\n"),sep="")
  }
  
  return(list("perf.classifier_on_raw"=perf.classifier_on_raw,"perf.classifier_on_autoencoder"=perf.classifier_on_autoencoder,"perf.ratio"=perf.ratio))   #return the classifier performances
}
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
### END OF FUNCTIONS ###

# Load the MNIST dataset:
load_mnist("./data/")  #train contains training images and labels, and test contains testing images and labels

# Display some training images:
n.grid <- 10
par(mfrow=c(n.grid,n.grid))
par(mar=c(0,0,0,0))
for (i in 1:n.grid^2){
  show_digit(train$x[i,]) 
}

### Hyperparameters:
n.training <- 10000   #size of the training set of images
n.testing <- 10000   #size of the testing set of images
# Hyperparameters of the autoencoder:
rho <- 0.1
beta <- 0.01
lambda <- 1e-5
n.hidden.units <- c(28*28, 17*17, 10)
hidden.layers <- length(n.hidden.units)

tmp <- NULL
for (i in 1:hidden.layers) tmp <- append(tmp,paste("n.units.",i,sep=""))
hp <- data.frame(matrix(append(c(n.training,rho,beta,lambda,hidden.layers),n.hidden.units), nrow=1, ncol=5+hidden.layers))
colnames(hp) <- c("n.training", "rho", "beta", "lambda", "hidden.layers", tmp)
cat("Hyperparameters of the run:\n")
print(hp)

# Train 2 classifiers: vanilla RF on raw pixel inputs, and hybrid model (RF on stacked autoencoder):
instances.train <- sample.int(n=nrow(train$x),size=n.training,replace = FALSE)   #define indices of the training instances
instances.validate <- sample.int(n=nrow(train$x),size=n.training,replace = FALSE)   #define indices of the training instances
instances.validate <- setdiff(instances.validate,instances.train)   #make sure no training examples are in the validation set
instances.test <- sample.int(n=nrow(test$x),size=n.testing,replace = FALSE)   #define indices of the testing instances

classifiers <- train_2classifiers.per.hp(instances.train, hp, verbose=TRUE, worker.id="")

evaluate_2classifiers.per.hp(classifiers, instances.validate, hp, measure=performance.measure, verbose=FALSE)
evaluate_2classifiers.per.hp(classifiers, instances.test, hp, measure=performance.measure, verbose=FALSE)
