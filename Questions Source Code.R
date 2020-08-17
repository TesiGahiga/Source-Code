
#Q1a
#setting up the array
row_total=9 ; col_total=9
square=data.frame(array(c(1:row_total), dim=c(row_total,col_total)))

#set up the target
x=1 ; y=1 ;  sum=square[x,y] ; target = 65

#Create the number of values for R and D we will need for this matrix and store them in a sequence. 
#Note:To reach the highest sum in square, we would have to go (1)1all the way down and then all the way right(i.e.DDDDDDRRRRRR). 
#(2)To have half the sum,the operations would need to alternate between R and D until the end of the matrix (i.e.RDRDRDRD...). 
#(3)To reach the lowest number, it will be all the way right then all the way down (i.e.RRRRRRDDDDDD) .

sequence=c(rep("D",row_total-1),rep("R",col_total-1))

#The for loop instructs R to go down (x+1) for every "D" and on the right for every "Right".
for (col in 1:100000000){ 
  x=1 ; y=1 ;  sum=square[x,y]
  for (i in 1:length(sequence)) {
    print(c(i,square[x,y]))
    if (sequence[i] == "R") {y=y+1}
    if (sequence[i] == "D") {x=x+1}
 #These sum and print commands instruct R to calculate the sum of the numbers it passed through with the entire sequence of "R" and "D" operations. 
  sum=sum+square[x,y]
  }
  print(c("sequence",sequence,"sum",sum))

  #The programme starts at the highest possible sum with the operations option (3), and reduces it by 1 until the target is reached. 
  #To do so, it goes through the sequence of operations for each sum from the highest possible sum, and identifies every time it encounters one "DR" operation sequence to swap it to "RD". 
 
  if (target == sum) {break}
  for (j in 1:(length(sequence)-1)){
    if (sequence[j] == "D" && sequence[j+1] == "R"){
      sequence[j] = "R" ; sequence[j+1] = "D"
      break
    }
  }
}
sequence
Target_65<-c(sum,sequence)
print(Target_65)

#Target 72
row_total=9 ; col_total=9
square=data.frame(array(c(1:row_total), dim=c(row_total,col_total)))

x=1 ; y=1 ;  sum=square[x,y] ; target = 72

sequence=c(rep("D",row_total-1),rep("R",col_total-1))
#
for (col in 1:100000000){ 
  x=1 ; y=1 ;  sum=square[x,y]
  for (i in 1:length(sequence)) {
    print(c(i,square[x,y]))
    if (sequence[i] == "R") {y=y+1}
    if (sequence[i] == "D") {x=x+1}
    sum=sum+square[x,y]
  }
  print(c("sequence",sequence,"sum",sum))
  
  if (target == sum) {break}
  for (j in 1:(length(sequence)-1)){
    if (sequence[j] == "D" && sequence[j+1] == "R"){
      sequence[j] = "R" ; sequence[j+1] = "D"
      break
    }
  }
}
sequence
Target_72<-c(sum,sequence)
print(Target_72)

#Target 90
row_total=9 ; col_total=9
square=data.frame(array(c(1:row_total), dim=c(row_total,col_total)))

x=1 ; y=1 ;  sum=square[x,y] ; target = 90

sequence=c(rep("D",row_total-1),rep("R",col_total-1))
#
for (col in 1:100000000){ 
  x=1 ; y=1 ;  sum=square[x,y]
  for (i in 1:length(sequence)) {
    print(c(i,square[x,y]))
    if (sequence[i] == "R") {y=y+1}
    if (sequence[i] == "D") {x=x+1}
    sum=sum+square[x,y]
  }
  print(c("sequence",sequence,"sum",sum))
  
  if (target == sum) {break}
  for (j in 1:(length(sequence)-1)){
    if (sequence[j] == "D" && sequence[j+1] == "R"){
      sequence[j] = "R" ; sequence[j+1] = "D"
      break
    }
  }
}
sequence
Target_90<-c(sum,sequence)
print(Target_90)

#Target 110
row_total=9 ; col_total=9
square=data.frame(array(c(1:row_total), dim=c(row_total,col_total)))

x=1 ; y=1 ;  sum=square[x,y] ; target = 110

sequence=c(rep("D",row_total-1),rep("R",col_total-1))
#
for (col in 1:100000000){ 
  x=1 ; y=1 ;  sum=square[x,y]
  for (i in 1:length(sequence)) {
    print(c(i,square[x,y]))
    if (sequence[i] == "R") {y=y+1}
    if (sequence[i] == "D") {x=x+1}
    sum=sum+square[x,y]
  }
  print(c("sequence",sequence,"sum",sum))
  
  if (target == sum) {break}
  for (j in 1:(length(sequence)-1)){
    if (sequence[j] == "D" && sequence[j+1] == "R"){
      sequence[j] = "R" ; sequence[j+1] = "D"
      break
    }
  }
}
sequence
Target_110<-c(sum,sequence)
print(Target_110)

print(Target_65)
print(Target_72)
print(Target_90)
print(Target_110)

#Q1b
row_total=90000 ; col_total=100000
square=data.frame(array(c(1:row_total), dim=c(row_total,col_total)))

x=1 ; y=1 ;  sum=square[x,y] ; target = 87127231192

sequence=c(rep("D",row_total-1),rep("R",col_total-1))

for (col in 1:1000000000000000000000000000000000000000000000000000){ 
  x=1 ; y=1 ;  sum=square[x,y]
  for (i in 1:length(sequence)) {
    print(c(i,square[x,y]))
    if (sequence[i] == "R") {y=y+1}
    if (sequence[i] == "D") {x=x+1}
    sum=sum+square[x,y]
  }
  print(c("sum",sum,"sequence",sequence))
  
  if (target == sum) {break}
  for (j in 1:(length(sequence)-1)){
    if (sequence[j] == "D" && sequence[j+1] == "R"){
      sequence[j] = "R" ; sequence[j+1] = "D"
      break
    }
  }
}
sequence
Target_1bi<-c(sum,sequence)
print(Target_1bi)

#Target 5995691682
row_total=90000 ; col_total=100000
square=data.frame(array(c(1:row_total), dim=c(row_total,col_total)))

x=1 ; y=1 ;  sum=square[x,y] ; target = 5995691682

sequence=c(rep("D",row_total-1),rep("R",col_total-1))

for (col in 1:10000000000000000000000000){ 
  x=1 ; y=1 ;  sum=square[x,y]
  for (i in 1:length(sequence)) {
    print(c(i,square[x,y]))
    if (sequence[i] == "R") {y=y+1}
    if (sequence[i] == "D") {x=x+1}
    sum=sum+square[x,y]
  }
  print(c("sequence",sequence,"sum",sum))
  
  if (target == sum) {break}
  for (j in 1:(length(sequence)-1)){
    if (sequence[j] == "D" && sequence[j+1] == "R"){
      sequence[j] = "R" ; sequence[j+1] = "D"
      break
    }
  }
}
sequence
Target_1bii<-c(sum,sequence)
print(Target_1bii)

max.print(Target_1bi)
max.print(Target_1bii)

#Q4
#//---------------Input---------------//
#Loading input data
input_question_4 <- read.delim("C:/Users/aflay/OneDrive/Desktop/input_question_4", header=FALSE)
View(input_question_4)
#Loading packages 
install.packages("imager")
install.packages("BiocManager")
BiocManager::install(version = "3.11")
BiocManager::install("EBImage")

#Convert input class to image  
class(input_question_4)
x <- data.matrix(input_question_4)

library("imager")
#as.cimg converts data (here, an array) into a cimg object. "Imager implements various converters that turn your data into cimg objects. If you convert from vector (which only has a length, and no dimension), either specify dimensions explicitly or some guesswork will be involved. See examples for clarifications"(Bartheleme S., 2020)
z <- as.cimg(x,v.name = "value", dims=c(20,10,1,1))

# Identify connected components in image
library(EBImage)
#bwlabel identifies the connected components in a binary image. "All pixels for each connected set of foreground (non-zero) pixels in x are set to an unique increasing integer, starting from 1. Hence, max(x) gives the number of connected objects in x."(Pau G et al., 2010)
y <- bwlabel(z)

#Convert image into a matrix to visualise connected components
Output_question_4 <- data.matrix(y)
print(Output_question_4)

#references: 
#Pau G, Fuchs F, Sklyar O, Boutros M, Huber W (2010). "EBImage-an R package for image processing with applications to cellular phenotypes." Bioinformatics, 26(7), 979-981. doi: 10.1093/bioinformatics/btq046.
#Barthelme S. (2020). imager: Image Processing Library Based on 'CImg'. R package


#Q5.1
L=5

#Store values "R" and "B" in square grid 
square=t(data.frame(array(c("R","B"), dim=c(L,L))))
#square[1,2]="R"

penalty=0

#This programme reads the "R" or "B" value in each each row and column 1 by 1 until row 4 and column 4. We seperate column 5 and row 5 as we do not want to compare their "R" or "B" values to their non-existent neughbors on the right and neighbors on the left respectively.
for (x in 1:(nrow(square)-1)) {
  for (y in 1:(ncol(square)-1)) {
    print(c(x,y))    
    
#Compares to the neighbor on the right and adds a penalty if they are the same color up until row 4 and column 4
    if (square[x,y] == square[x+1,y]) {penalty=penalty+1}
  }}


#Compares to neighbor below  and adds a penalty if they are the same color up until row 4 and column 4
if (square[x,y] == square[x,y+1]) {penalty=penalty+1}


#Compares to neighbor below for column 5 and adds a penalty if they are the same color
x = 5
for (y in 1:(ncol(square)-1)) {
  print(c(x,y))   
  if (square[x,y] == square[x,y+1]) {penalty=penalty+1}
}

#Compares to neighbor below for row 5 and adds a penalty if they are the same color
y=5
for (x in 1:(nrow(square)-1)) {
  print(c(x,y))
  if (square[x,y] == square[x+1,y]) {penalty=penalty+1}
}
penalty
print(square)

#Q5.2
L=64
#stack all all the coloured beads in order
square=c(rep("R",139),rep("B",1451),rep("G",977),rep("W",1072),rep("Y",457))

#randomise the beads' position. This will decrease the likelihood of a penalty.  
square=square[sample(length(square))]

#put the beads in an array
square2=array(square,dim=c(L,L))


penalty=0

for (x in 1:(nrow(square2)-1)) {
  for (y in 1:(ncol(square2)-1)) {
    print(c(x,y))    
    
  #Compares to the neighbour on the right up until row 63 and column 63. See 5.1. for rationale.
    if (square2[x,y] == square2[x+1,y]) {penalty=penalty+1}
  }}


#Compares to neighbour below  up until coordinates until row 63 and column 63
if (square2[x,y] == square2[x,y+1]) {penalty=penalty+1}


#Compares to neighbour below for column 64
x = L
for (y in 1:(ncol(square2)-1)) {
  print(c(x,y))   
  if (square2[x,y] == square2[x,y+1]) {penalty=penalty+1}
}

#Compares to neighbour below for row 64
y=L
for (x in 1:(nrow(square2)-1)) {
  print(c(x,y))
  if (square2[x,y] == square2[x+1,y]) {penalty=penalty+1}
}
penalty
write.csv(square2)

#Q6
#Using the even-odd algorithm or the Jordan algorithm to solve this Point in Polygon problem (Algorithms and Technologies, 2019)
#Start of the algorithm
point_in_polygon <- function(polygon, point){
  #' Raycasting Algorithm to find out whether a point is in a given polygon.
  #' Performs the even-odd-rule Algorithm to find out whether a point is in a given polygon.
  #' This runs in O(n) where n is the number of edges of the polygon.
  #' @param polygon an array representation of the polygon where polygon[i,1] is the x Value of the i-th point and polygon[i,2] is the y Value.
  #' @param point   an array representation of the point where point[1] is its x Value and point[2] is its y Value
  #' @return whether the point is in the polygon (not on the edge, just turn < into <= and > into >= for that)
  
  # A point is in a polygon if a line from the point to infinity crosses the polygon an odd number of times
  odd = FALSE
  # For each edge (In this case for each point of the polygon and the previous one)
  i = 0
  j = nrow(polygon) - 1
  while(i < nrow(polygon) - 1){
    i = i + 1
    # If a line from the point into infinity crosses this edge
    # One point needs to be above, one below our y coordinate
    # ...and the edge doesn't cross our Y corrdinate before our x coordinate (but between our x coordinate and infinity)
    if (((polygon[i,2] > point[2]) != (polygon[j,2] > point[2])) 
        && (point[1] < ((polygon[j,1] - polygon[i,1]) * (point[2] - polygon[i,2]) / (polygon[j,2] - polygon[i,2])) + polygon[i,1])){
      # Invert odd
      odd = !odd
    }
    j = i
  }
  # If the number of crossings was odd, the point is in the polygon
  return (odd)
}
#End of algorithm

#store the coordinates for the points and the polygon
par(mar=c(5.1,4.1,4.1,2.1)) 
shape=cbind(c(4,2,3,2,5,9,14,20,18,11),c(3,6,12,17,20,21,19,14,3,7))
points=cbind(c(7,10,11,12,16,16,17,18,18,20),c(11,14,4,21,3,10,4,7,17,7))

#Set up array of 2 elements, inside outside. 
#point in polygon gives you 0 or 1 based on if the points are inside or outside. By adding 1, and creating an array of two elements "inside" and "outside", we are then able to print whether the points are 1/inside or 2/outside.

for (i in 1:nrow(points)){
  print(c(points[i,],c("Outside","Inside")[point_in_polygon(shape,points[i,])+1]))
}
#plot the polygon to double check if the points are indeed inside or outside 
plot(1,1, col = "white",xlim=c(0,max(shape[,1])),ylim=c(0,max(shape[,2])),axes = TRUE)
polygon(x = shape[,1],y = shape[,2],col = "#1b98e0",border = "red",lwd = 5)  
lines(points[,1],points[,2],col="black",type = "p")
for (i in 1:nrow(points)){
  print(c(points[i,],c("Outside","Inside")[point_in_polygon(shape,points[i,])+1]))
}

#Reference:
#Algorithms and Technologies(2019)"Point in Polygon in R". Retrieved from https://www.algorithms-and-technologies.com/point_in_polygon/r
 
#Q7.1.a
x <- x coordinates
y <- y coordinates
m <- L1
n <- L2

one_D_index = x  + y*m

#Q.7.1.b.i.
#Coordinates to Index
Coordinates<- read.delim("C:/Users/aflay/OneDrive/Desktop/input_coordinates_7_1.txt", header=TRUE)

x <- Coordinates[,1]
y <- Coordinates[,2]
m <- 50
n <- 57

one_D_index = x  + y*m
View(one_D_index)

#Q7.1.b.ii
#Index to Coordinates
Seven_One_Index <- unlist(read.delim("C:/Users/aflay/OneDrive/Desktop/input_index_7_1.txt",header=TRUE))
m <- 50
n <- 57
two_D_index = cbind(Seven_One_Index%%m,(as.integer(Seven_One_Index/m)))
print("index"=Seven_One_Index,"x"=two_D_index[,1],"y"=two_D_index[,2])
write.csv(cbind("index"=Seven_One_Index,"x"=two_D_index[,1],"y"=two_D_index[,2]))

#Q7.2.b

Coordinates_Seven_Two <- data.frame(read.delim("C:/Users/aflay/OneDrive/Desktop/Question 7.2/input_coordinates_7_2.txt", header=TRUE))

#added a constant of 1 and of 0 to have elements to refer to when the loop works on the first coordinate. 
#This programme works with 7 dimensional grid for this exercise, however it doesn't change the original grids' properties.

Coordinates_Seven_Two = cbind(1,Coordinates_Seven_Two)
L = c(1,4,8,5,9,6,7)

#This is making an array with the same number of elements as number of coordinates to store index
one_D_index=rep(0,nrow(Coordinates_Seven_Two))

#This loop will go through each set of coordinates set one at a time it will multiply all the other constants to the coordinates
#one_D_index[j] = one_D_index[j] is necessary to ensure that he prior calculations are included in the total sum as the dimensions are calculated one at a time. 
#prod(L[1:(i-1)]) multiplies the coordinates with all the other constants in the dimensions prior to itself

for (j in 1:nrow(Coordinates_Seven_Two)){
  #i=1; j=1;{
  for (i in 1:length(L)){
    one_D_index[j] = one_D_index[j] + Coordinates_Seven_Two[j,i]*prod(L[1:(i-1)])
  }
}
write.csv(cbind(Coordinates_Seven_Two,one_D_index))

#Q8.1

#At initial condtions:
#dS/dt= k2[ES] - k1[S][E],                    [S](0) = S0

#dE/dt = (k2 + k3)[ES] - k1[S][E],            [E](0) = E0

#dES/dt= k1[S][E] - (k2 + k3)[ES],            [SE](0) = 0

#dP/dt= k3[ES],                               [P](0) = 0.


#Q8.2

k1<-100
k2<-600
k3<-150

Et <- 1
S0<-10 


#x1 <-S
#x2 <- ES
#x3 <- P

#S function:
#y=k2*x2 - k1*S0*Et

#E function:
#y=(k2 + k3)*x2 - k1*x1*Et

#ES function
#y=k1*x1*Et - (k2+k3)*x2

#P function
#y=k3*x2

#Installing packages
#The Rmutil package is "a toolkit of functions for nonlinear regression and repeated measurements not to be used by itself but called by other Lindsey packages such as 'gnlm', 'stable', 'growth', 'repeated', and 'event'"Lindsey J., Swihart B.,2020).
install.packages("rmutil")
library(rmutil)

#At initial conditions, S^t=0 = So; E^t=0 = Et; ES^t=0 = 0; P^t=0 = 0

#Solving the 4 equations with using the fourth-order Runge-Kutta method :
#S:
fnS <- function(y,x) k2*x2 - k1*S0*Et
soln <- runge.kutta(fnS,10,)

#ES:
fnES <- function(y,x) y=k1*x1*Et - (k2+k3)*x2
soln <- runge.kutta(fnES,10,)

#E:
fnE <- function(y,x) (k2 + k3)*x2 - k1*x1*Et
soln <- runge.kutta(fnE,10,)


#P:
fnP <- function(y,x) k3*x2 
soln <- runge.kutta(fnP,10,)


#References
#Lindsey J., Swihart B. (2020) "rmutil: Utilities for Nonlinear Regression and Repeated Measurements Models" CRAN R-Project. https://cran.r-project.org/web/packages/rmutil/index.html

#Q8.3
#At initial conditions, [S]^t=0 = So; [E]^t=0 = Et; [ES]^t=0 = 0; [P]^t=0 = 0
#Et = E + ES

#calculating Km
#Km=(k2+k3)/k1=7.5

#Michaelis-Menten Equation
#V=(Vmax*S)/(Km+S)
#Where y=V and x=S

#y=(150*x)/(7.5+x)

#The lattice package is "a powerful and elegant high-level data visualization system inspired by Trellis graphics, with an emphasis on multivariate data. Lattice is sufficient for typical graphics needs, and is also flexible enough to handle most nonstandard requirements." (
install.packages(lattice)
Library(lattice)
Velocity_V= function(x) { (150*x)/(7.5+x)}
Substrate_S= 1:100

#xyplot produces bivariate scatterplots or time-series plots
xyplot(Velocity_V(Substrate_S)~Substrate_S,type="l")

#Vmax. Reaction is fastest when all enzymes are complexed with S. Based on the graph, we can see that Vmax is at 150. We can also verify this by calculating it:
#Vmax=k3*Et = 150
#Alternatively, we know that When Km=S, V0=1/2Vmax.We can see on the graph that when S=7.5, V0=75, so Vmax=75*2=150


#Reference:
#Sarkar D.,  Andrews F.,Wright K., Klepeis N., Murrell P. (2020) "Package 'lattice' ", CRAN