# Load libraries:
set.seed(123)

require(abind)
require(EBImage)
require(mxnet)

# set working directory
setwd("/Users/masimonson/Documents/Code/R/R_2017/MxNetExample/ImageClassification/train")


## define function to create image array required by mxnet
# inputs: images = vector of image file names, w=width in pixels, h=height in pixels
# 
# output: array containing images data in format expected by mxnet
createImgArray <- function(images,w,h){
  images <- as.vector(images)
  imgArray <- array(dim = c(w, h, 1, 0))
  # loop resize images and set them to greyscale and store in array format expected by mxnet
  for(i in 1:length(images)){
    # Image name
    imgname <- images[i]
    # Read image
    img <- readImage(imgname)
    # Resize image 
    img_resized <- resize(img, w = w, h = h)
    # Set to grayscale
    grayimg <- channel(img_resized,"gray")
    imgArrayTemp <- array(data = grayimg,dim = c(w, h, 1, 1))
    imgArray <- abind(imgArray, imgArrayTemp)
    print(i)
  }
  return(imgArray)
}

# matrix containing names of all image files
images.frame <- data.frame(matrix(list.files(pattern = "*jpg"),ncol=1))
names(images.frame) <- "img"
images.frame$index <- 1:nrow(images.frame)
images.frame <- cbind.data.frame(images.frame,c(rep(1, 12500),rep(0, 12500)))
names(images.frame) <- c("img","index","category")

# generate training/test set split:
train.index <- sample.int(nrow(images.frame),0.7*nrow(images.frame),replace=FALSE)
train.images.frame <- images.frame[train.index,]
test.images.frame <- images.frame[-train.index,]
  
# create array for training set and create vector containing class labels

# Set width
w <- 72
# Set height
h <- 72
train.Array <- createImgArray(train.images.frame$img,w,h) # training array containing image data
train.class <- train.images.frame$category # training labels


test.Array <- createImgArray(test.images.frame$img,w,h) # testing array containing image data
test.class <- test.images.frame$category # testing labels


#Define mxnet model
data <- mx.symbol.Variable('data')

# 1st convolutional layer 5x5 kernel and 20 filters.
conv_1 <- mx.symbol.Convolution(data= data, kernel = c(5,5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data= conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2,2), stride = c(2,2))

# 2nd convolutional layer 5x5 kernel and 50 filters.
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5,5), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data = tanh_2, pool_type = "max", kernel = c(2,2), stride = c(2,2))

# 1st fully connected layer
flat <- mx.symbol.Flatten(data = pool_2)
fcl_1 <- mx.symbol.FullyConnected(data = flat, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fcl_1, act_type = "tanh")

# 2nd fully connected layer
fcl_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 2)
# Output
NN_model <- mx.symbol.SoftmaxOutput(data = fcl_2)

# Set seed for reproducibility
mx.set.seed(100)

# Device used. 
device <- mx.gpu()

# Train 
model <- mx.model.FeedForward.create(NN_model, X = train.Array, y = train.class,
                                     ctx = device,
                                     num.round = 10,
                                     array.batch.size = 100,
                                     learning.rate = 0.05,
                                     momentum = 0.9,
                                     wd = 0.00001,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

# Test
predict_probs <- predict(model, test.Array)
predicted_labels <- max.col(t(predict_probs)) - 1
table(test.class, predicted_labels)
accuracy <- sum(diag(table(test.class, predicted_labels)))/sum(table(test.class, predicted_labels))
