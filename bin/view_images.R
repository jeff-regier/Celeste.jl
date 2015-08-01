library(ggplot2)
library(dplyr)
library(reshape2)
library(gridExtra)


######### Looking at a single stamp: 
images <- list()
for (b in 1:5) {
  file_name <- sprintf("/tmp/pixels_%d.csv", b)
  print(file_name)
  du <- as.matrix(read.csv(file_name, header=F))
  colnames(du) <- 1:ncol(du)
  rownames(du) <- 1:nrow(du)
  du_melt <- melt(du)
  names(du_melt) <- c("row", "col", "value")
  du_melt$band <- b
  images[[as.character(b)]] <- du_melt
}
du <- do.call(rbind, images)

ggplot(du) +
  geom_raster(data=du, aes(x=row, y=col, fill=value)) +
  scale_fill_continuous(low="#000000", high="#FFFFFF", na.value = "purple") +
  facet_grid(band ~ .)

########## Looking at a whole image:
dcat <- read.csv("/tmp/catalog-003900-6-0269.csv", header=T)

band <- 3
band.letters <- c("u", "g", "r", "i", "z")
file_name <- sprintf("/tmp/frame-%s-003900-6-0269.csv", band.letters[band])
du <- as.matrix(read.csv(file_name, header=F))
colnames(du) <- 1:ncol(du)
rownames(du) <- 1:nrow(du)
du_melt <- melt(du)
names(du_melt) <- c("row", "col", "value")

band.letter <- band.letters[band]
cat.px.x.col <- sprintf("pix_x_%d", band)
cat.px.y.col <- sprintf("pix_y_%d", band)

img_xlim <- c(150, 350)
img_ylim <- c(300, 450)
this_cat <- filter(dcat,
                  dcat[[cat.px.x.col]]  >= img_xlim[1],
                  dcat[[cat.px.x.col]]  <= img_xlim[2],
                  dcat[[cat.px.y.col]]  >= img_ylim[1],
                  dcat[[cat.px.y.col]]  <= img_ylim[2])
this_cat$x <- this_cat[[cat.px.x.col]]
this_cat$y <- this_cat[[cat.px.y.col]]

this_du <- filter(du_melt,
                  du_melt[["row"]]  >= img_xlim[1],
                  du_melt[["row"]]  <= img_xlim[2],
                  du_melt[["col"]]  >= img_ylim[1],
                  du_melt[["col"]]  <= img_ylim[2])

grid.arrange(
  ggplot() +
    geom_raster(data=this_du, aes(x=row, y=col, fill=value)) +
    geom_point(data=this_cat, aes(x=x, y=y, color=is_star), size=5) +
    scale_fill_continuous(low="#000000", high="#FFFFFF", na.value = "purple")
  ,
  ggplot() +
    geom_raster(data=this_du, aes(x=row, y=col, fill=value)) +
    scale_fill_continuous(low="#000000", high="#FFFFFF", na.value = "purple")
  ,
  ncol=2
)
