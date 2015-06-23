library(ggplot2)
library(dplyr)
library(reshape2)
library(gridExtra)

d <- as.matrix(read.csv("/tmp/frame-u-003900-6-0269.csv", header=F))

dcat <- read.csv("/tmp/catalog-u-003900-6-0269.csv", header=T)

image(d[1:100, 1:100])
