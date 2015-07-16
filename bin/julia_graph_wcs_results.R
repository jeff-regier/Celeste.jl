# A temporary helper file to graph things while PyPlot is broken.

library(ggplot2)
library(dplyr)
library(gridExtra)
library(reshape2)

ReadJuliaCSV <- function(filename) {
  data <- read.csv(filename, header=F, sep=",", colClasses="numeric")
  data <- do.call(rbind, data)
  colnames(data) <- 1:ncol(data)
  rownames(data) <- 1:nrow(data)
  data_melt <- melt(data)
  names(data_melt) <- c("row", "col", "value")
  return(data_melt)  
}

images <- data.frame()
psfs <- data.frame()
for (b in 1:5) {
  
  raw_psf <- ReadJuliaCSV(sprintf("/tmp/raw_psf_%d.csv", b))
  fit_psf <- ReadJuliaCSV(sprintf("/tmp/fit_psf_%d.csv", b))
  pixels <- ReadJuliaCSV(sprintf("/tmp/pixels_%d.csv", b))
  synth_pixels <- ReadJuliaCSV(sprintf("/tmp/synth_pixels_%d.csv", b))
  e_image <- ReadJuliaCSV(sprintf("/tmp/e_image_%d.csv", b))
  
  pixels$image <- "pixels"
  synth_pixels$image <- "synthetic"
  e_image$image <- "fit"
  pixels$b <- synth_pixels$b <- e_image$b <- raw_psf$b <- fit_psf$b <- b
  
  raw_psf$image <- "raw"
  fit_psf$image <- "fit"
  
  images <- rbind(images, rbind(pixels, synth_pixels, e_image))  
  psfs <- rbind(psfs, raw_psf, fit_psf)
}
images2 <- dcast(images, row + col + b ~ image)
psfs2 <- dcast(psfs, row + col + b ~ image)

ggplot(images) +
  geom_tile(aes(x=row, y=col, fill=value, group=image)) +
  facet_grid(image ~ b)

scale <- max(images2$pixels)
grid.arrange(
  ggplot(images2) + geom_tile(aes(x=row, y=col, fill=pixels, group=b)) +
    scale_fill_continuous(low="#000000", high="#FFFFFF", na.value = "red",
                          limits=c(-scale, scale)) +
    facet_grid(b ~ .) + ggtitle("Image")
  ,
  ggplot(images2) + geom_tile(aes(x=row, y=col, fill=pixels - fit, group=b)) +
    scale_fill_continuous(low="#000000", high="#FFFFFF", na.value = "red",
                          limits=c(-scale, scale)) +
    facet_grid(b ~ .) + ggtitle("Image - fit")
  ,
  ggplot(images2) + geom_tile(aes(x=row, y=col, fill=pixels - synthetic, group=b)) +
    scale_fill_continuous(low="#000000", high="#FFFFFF", na.value = "red",
                          limits=c(-scale, scale)) +
    facet_grid(b ~ .) + ggtitle("Image - synthetic")
  ,
  ncol=3
)

ggplot(filter(psfs2, row >= 20, row <= 31, col >= 20, col <= 31)) +
  geom_raster(aes(x=row, y=col, fill=raw - fit)) + ggtitle("PSF difference")

grid.arrange(
  ggplot(filter(raw_psf, row >= 20, row <= 31, col >= 20, col <= 31)) +
    geom_raster(aes(x=row, y=col, fill=value)) + ggtitle("Raw PSF")
  ,
  ggplot(filter(fit_psf, row >= 20, row <= 31, col >= 20, col <= 31)) +
    geom_raster(aes(x=row, y=col, fill=value)) + ggtitle("Fit PSF")
  ,
  ncol=2
)

