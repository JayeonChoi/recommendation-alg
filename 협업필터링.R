########Project
install.packages("data.table")
library(data.table)
customer = fread("L:/project/bigdata/customer.txt")
purchase = fread("L:/project/bigdata/purchase.txt")
goods = fread("L:/project/bigdata/goods.txt")
others = fread("L:/project/bigdata/others.txt")
membership = fread("L:/project/bigdata/membership.txt")
channel = fread("L:/project/bigdata/channel.txt")
colnames(customer)= c("id", "gender", "age", "resi")
colnames(others) = c("id", "comp","other","date")
colnames(membership) = c("id", "member","edate")

head(others)


head(membership)
sink('out.txt')
closeAllConnections()
gc(reset=T)

head(purchase)
head(goods)
head(others)
head(membership)
head(channel)

str(channel)

#데이터 탐색
simplepur = purchase[,c(5,6)]
colnames(simplepur) <- c("product", "ID")

head(simplepur, 100)

all(is.na(simplepur))
length(unique(simplepur$ID)) == length(customer$고객번호)
length(unique(simplepur$product))  == length(goods$소분류코드)


length(unique(simplepur$ID))
length(unique(simplepur$product))
table(purchase$제휴사) 
table(substr(goods$소분류코드,1,1))
table(simplepur$ID)[1:5,]

#바이너리 wide 포맷형태로 재구성
simplepur[,value:=1]
simplepur = simplepur[order(simplepur$ID),]
head(simplepur)

k=simplepur[1:max(which(simplepur$ID==1000)),]
k2=simplepur[(max(which(simplepur$ID==1000))+1):max(which(simplepur$ID==2000)),]
k3=simplepur[(max(which(simplepur$ID==2000))+1):max(which(simplepur$ID==3000)),]
k4=simplepur[(max(which(simplepur$ID==3000))+1):max(which(simplepur$ID==4000)),]
k5=simplepur[(max(which(simplepur$ID==4000))+1):max(which(simplepur$ID==5000)),]
k6=simplepur[(max(which(simplepur$ID==5000))+1):max(which(simplepur$ID==6000)),]
k7=simplepur[(max(which(simplepur$ID==6000))+1):max(which(simplepur$ID==7000)),]
k8=simplepur[(max(which(simplepur$ID==7000))+1):max(which(simplepur$ID==8000)),]
k9=simplepur[(max(which(simplepur$ID==8000))+1):max(which(simplepur$ID==9000)),]
k10=simplepur[(max(which(simplepur$ID==9000))+1):max(which(simplepur$ID==10000)),]
k11=simplepur[(max(which(simplepur$ID==10000))+1):max(which(simplepur$ID==11000)),]
k12=simplepur[(max(which(simplepur$ID==11000))+1):max(which(simplepur$ID==12000)),]
k13=simplepur[(max(which(simplepur$ID==12000))+1):max(which(simplepur$ID==13000)),]
k14=simplepur[(max(which(simplepur$ID==13000))+1):max(which(simplepur$ID==14000)),]
k15=simplepur[(max(which(simplepur$ID==14000))+1):max(which(simplepur$ID==15000)),]
k16=simplepur[(max(which(simplepur$ID==15000))+1):max(which(simplepur$ID==16000)),]
k17=simplepur[(max(which(simplepur$ID==16000))+1):max(which(simplepur$ID==17000)),]
k18=simplepur[(max(which(simplepur$ID==17000))+1):max(which(simplepur$ID==18000)),]
k19=simplepur[(max(which(simplepur$ID==18000))+1):max(which(simplepur$ID==19000)),]
k20=simplepur[(max(which(simplepur$ID==19000))+1):length(simplepur$ID),]

pur_bin = reshape(data=k, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin2 = reshape(data=k2, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin3 = reshape(data=k3, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin4 = reshape(data=k4, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin5 = reshape(data=k5, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin6 = reshape(data=k6, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin7 = reshape(data=k7, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin8 = reshape(data=k8, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin9 = reshape(data=k9, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin10 = reshape(data=k10, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin11 = reshape(data=k11, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin12 = reshape(data=k12, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin13 = reshape(data=k13, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin14 = reshape(data=k14, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin15 = reshape(data=k15, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin16 = reshape(data=k16, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin17 = reshape(data=k17, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin18 = reshape(data=k18, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin19 = reshape(data=k19, direction = "wide", idvar="ID", timevar = "product", v.names="value")
pur_bin20 = reshape(data=k20, direction = "wide", idvar="ID", timevar = "product", v.names="value")

#쪼개서 계산
g = merge(x = pur_bin, y = pur_bin2, by=union(colnames(pur_bin), colnames(pur_bin2)), all = TRUE)
g2 = merge(x = pur_bin3, y=pur_bin4, all=TRUE)
g3 = merge(x = pur_bin5, y=pur_bin6, all=TRUE)
g4 = merge(x = pur_bin7, y=pur_bin8, all=TRUE)
g5 = merge(x = pur_bin9, y=pur_bin10, all=TRUE)
g6 = merge(x = pur_bin11, y=pur_bin12, all=TRUE)
g7 = merge(x = pur_bin13, y=pur_bin14, all=TRUE)
g8 = merge(x = pur_bin15, y=pur_bin16, all=TRUE)
g9 = merge(x = pur_bin17, y=pur_bin18, all=TRUE)
g10 = merge(x = pur_bin19, y=pur_bin20, all=TRUE)

if (length(g10.toAdd <- setdiff (colnames(g), names(g10))))
  g10[, c(g10.toAdd) := NA]
dim(g10)

if (length(g.toAdd <- setdiff (names(g9), names(g))))
  g[, c(g.toAdd) := NA]


pur_bin.t = rbind(g, g2, g3, g4,g5,g6,g7,g8,g9,g10)

dim(pur_bin.t)

#횟수고려
install.packages("sqldf")
library(sqldf)
dup <- sqldf("SELECT Product, ID, COUNT(*)
             FROM simplepur
             GROUP BY ID, PRODUCT")
colnames(dup)[3] = "number"
head(dup)
dim(dup)
w1=dup[1:max(which(dup$ID==4000)),]
w2=dup[(max(which(dup$ID==4000))+1):max(which(dup$ID==8000)),]
w3=dup[(max(which(dup$ID==8000))+1):max(which(dup$ID==12000)),]
w4=dup[(max(which(dup$ID==12000))+1):max(which(dup$ID==16000)),]
w5=dup[(max(which(dup$ID==16000))+1):length(dup$ID),]

du1 = reshape(data=w1, direction = "wide", idvar="ID", timevar="product", v.names="number")
du2 = reshape(data=w2, direction = "wide", idvar="ID", timevar="product", v.names="number")
du3 = reshape(data=w3, direction = "wide", idvar="ID", timevar="product", v.names="number")
du4 = reshape(data=w4, direction = "wide", idvar="ID", timevar="product", v.names="number")
du5 = reshape(data=w5, direction = "wide", idvar="ID", timevar="product", v.names="number")

m1 = merge(x = du1, y = du2, all = TRUE)
m2 = merge(x = du3, y=du4, all=TRUE)
m3 = merge(x = data.frame(du5), y=data.frame(m1), all=TRUE)
m4 = merge(x = data.frame(m1), y=data.frame(m2), all=TRUE)
m5 = merge(x = data.frame(m4), y=data.frame(du5), all=TRUE)
m5[1:5,1:5]


m5[,ID:=NULL]
setnames(x=m5,
         old=names(m5), new=substring(names(m5),8))
rownames(m5) = custom

rating.matrix=as.matrix(m5)
rating.matrix[is.na(rating.matrix)] = 0
rownames(rating.matrix) = custom

k = as.matrix(rating.matrix)
class(rating.matrix)
write.csv(rating.matrix, "number.csv", row.names = F)

library(recommenderlab)

num = read.csv("number.csv", stringsAsFactors=F)
rownames(num) = rownames
rmatrix = as(num, "realRatingMatrix")
image(rmatrix)
#미완성..
####

m = read.csv("hahahaha.csv", stringsAsFactors = F)
m=m[,-1]
rownames = customer$고객번호
rownames(m) = rownames
class(m)
m2 = as("matrix", m)
bin.rmatrix = as(m2, "binaryRatingMatrix")


####
custom = customer$고객번호
pur_bin.t[,ID:=NULL]
setnames(x=pur_bin.t,
         old=names(pur_bin.t), new=substring(names(pur_bin.t),7))
rownames(pur_bin.t) = custom
bin.matrix=as.matrix(pur_bin.t)
bin.matrix[is.na(bin.matrix)] = 0
head(bin.matrix)[,1:5]
write.csv(bin.matrix, "hahahaha.csv", row.names = F) #여기까지.

####
#구매한 횟수도 매트릭스로 본다면..?
rowCounts(bin.rmatrix)[1]

install.packages("recommenderlab")
library(recommenderlab)
bin.rmatrix = as(m2, "binaryRatingMatrix")

#시각화
install.packages("ggplot2")
library(ggplot2)
image(bin.rmatrix[1:500,1:500], main="이진 평점 매트릭스 시각화")#평점 매트릭스 
image(rmatrix[1:100, 1:100])

qplot(colCounts(bin.rmatrix)) + stat_bin(binwidth = 50) + ggtitle("상품별 구매횟수 분포표")
#대부분 아이템이 0~50번 정도 구매됨. 왜도가 매우 심함
qplot(rowCounts(bin.rmatrix)) + stat_bin(binwidth = 50) + ggtitle("고객별 구매횟수 분포표")
#골고루 분포
boxplot(rowCounts(bin.rmatrix))
title("고객 구매횟수 분포 boxplot")
boxplot(colCounts(bin.rmatrix))
title("상품별 판매횟수 분포 boxplot")

max(colCounts(bin.rmatrix))
min(colCounts(bin.rmatrix))
summary(colCounts(bin.rmatrix))
summary(rowCounts(bin.rmatrix))

#가장 많이 구매된 상품은 12372번, 1번만 구매된 상품도 있음.
#극단적인 상품 제외한 후 시각화(50번 이상구매된 상품, 50번 이상 구매한 고객)
n=colCounts(bin.rmatrix)
qplot(n[n<3000]) + stat_bin(binwidth = 30) + ggtitle("극단값 제거 상품구매 분포표")
bin.rmatrix <- bin.rmatrix[, n>=30] #최소 30번 이상 구매된 상품만 추천
sum(rowCounts(bin.rmatrix)==0) #그 아이템들만 구매했었던 손님. 0명인거 확인

dim(bin.rmatrix[,n<=20])

max(rowCounts(bin.rmatrix))
min(rowCounts(bin.rmatrix))
#1번만 구매한 손님, 최대 939번 구매한 손님 존재
#두가지 옵션-적게 구매한 손님 제외, 제외x
#제외하는 경우
bin.rmatrix.exc = bin.rmatrix[rowCounts(bin.rmatrix)>=50,]

#데이터분할+모델 만들고 평가(분할별로 1개씩)
data.set1 = evaluationScheme(data=bin.rmatrix.exc, method="split", train=0.7, given=50, goodRating=1, k=1)
recommnder1 = Recommender(data=getData(data.set1, "train"), method="IBCF", parameter=list(method="Jaccard"))
prediction1 = predict(recommnder1, newdata=getData(data.set1, "known"), n=3, type="topNList")
accuracy1 = calcPredictionAccuracy(x=prediction1, data=getData(data.set1, "unknown"), byUser=F, given=50)
accuracy1[5:7] #정확도 0.75, 재현력 0.012

data.set2 = evaluationScheme(data=bin.rmatrix.exc, method="bootstrap", train=0.7, given=50, goodRating=1, k=1)
recommnder2 = Recommender(data=getData(data.set2, "train"), method="IBCF", parameter=list(method="Jaccard", k=30))
prediction2 = predict(recommnder2, newdata=getData(data.set2, "known"), n=3, type="topNList")
accuracy2 = calcPredictionAccuracy(x=prediction2, data=getData(data.set2, "unknown"), byUser=F, given=50)
accuracy2[5:7] #정확도 0.74, 재현력 0.0119

data.set3 = evaluationScheme(data=bin.rmatrix.exc, method="cross-validation", k=4, train=0.7, given=50, goodRating=1)
recommender = Recommender(data=getData(data.set3, "train"), method="IBCF",  parameter=list(method="Jaccard", k=30))
prediction = predict(recommender, newdata=getData(data.set3, "known"), n=3, type="topNList")
accuracy = calcPredictionAccuracy(x=prediction, data=getData(data.set3, "unknown"), byUser=F, given=50)
accuracy #정확도 0.75정도.

data.set4 = evaluationScheme(data=bin.rmatrix.exc, method="bootstrap", train=0.8, given=50, goodRating=1, k=1)

#UBCF
recommnder11 = Recommender(data=getData(data.set1, "train"), method="UBCF", parameter=list(method="Jaccard"))
prediction11 = predict(recommnder11, newdata=getData(data.set1, "known"), n=3, type="topNList")
accuracy11 = calcPredictionAccuracy(x=prediction11, data=getData(data.set1, "unknown"), byUser=F, given=50)

memory.limit(12000)
recommnder22 = Recommender(data=getData(data.set2, "train"), method="UBCF", parameter=list(method="Jaccard"))
prediction22 = predict(recommnder22, newdata=getData(data.set2, "known"), n=3, type="topNList")
accuracy22 = calcPredictionAccuracy(x=prediction22, data=getData(data.set2, "unknown"), byUser=F, given=50)

recommender33 = Recommender(data=getData(data.set3, "train"), method="UBCF",  parameter=list(method="Jaccard"))
prediction33 = predict(recommender33, newdata=getData(data.set3, "known"), n=3, type="topNList")
accuracy33 = calcPredictionAccuracy(x=prediction33, data=getData(data.set3, "unknown"), byUser=F, given=50)

accuracy11[5:6]
accuracy22[5:6]
accuracy33[5:6]
c = rbind(accuracy11[5:6],accuracy22[5:6],accuracy33[5:6])
rownames(c) = c("split", "bootstrap", "k-fold")
plot(c, col=as.integer(factor(rownames(c))))
text(locator(1), "split")
text(locator(1), "bootstrap")
text(locator(1), "k-fold")

#최적화

data.set1 = evaluationScheme(data=bin.rmatrix.exc, method="split", train=0.7, given=50, goodRating=1, k=1)
list = list(IBCF = list(name="IBCF", param=list(method="Jaccard")),
            UBCF = list(name="UBCF", param=list(method="Jaccard")))                  
result = evaluate(x=data.set1, method = list, n=c(1,3,5,seq(10,100,10)))
plot(result, annotate=1, legend="topleft"); title("ROC Curve")
plot(result, "prec/rec", annotate=1, legend="bottomright"); title("Precision-recall") 

#월등히 UBCF가 좋음


#지금까지 최적모델 탐구. 이제 매개변수 최적화

vec_n = c(5,10,15,25,30,40)
mod.eval = lapply(vec_n, function(n){
  list(name="UBCF", param=list(method="Jaccard", nn=n))
})
names(mod.eval) = paste0("UBCF_n", vec_n)
n_recom = c(1,3,5,seq(10,100,10))
list_result = evaluate(x=data.set1, method=mod.eval, n=n_recom)
plot(list_result, annotate=1, legend="topleft"); title("ROC Curve")
plot(list_result, "prec/rec", annotate=1, legend="bottomright"); title("Precision-recall")

  
##최종모델
recommender = Recommender(data=getData(data.set3, "train"), method="UBCF",  parameter=list(method="Jaccard", nn=30))
prediction = predict(recommender33, newdata=getData(data.set3, "known"), n=3, type="topNList")
accuracy = calcPredictionAccuracy(x=prediction33, data=getData(data.set3, "unknown"), byUser=F, given=50)

as(prediction, "matrix")
r = getList(prediction, decode=T, rating=T)
r
result = sapply(prediction@items, function(x) prediction@itemLabels[x])
#user ID는 어디에..?
length(result)
result[[142]]



tf = sample(x=c(TRUE, FALSE), size=nrow(bin.rmatrix.exc), replace=T, prob=c(0.8,0.2))
data.train = bin.rmatrix.exc[tf, ]
data.test = bin.rmatrix.exc[!tf, ]

which = sample(x=1:5, size = nrow(bin.rmatrix.exc), replace=T)
result = list()
for(i in 1:5) {
  test <- which==i
  data.train = bin.rmatrix.exc[!test, ]
  data.test = bin.rmatrix.exc[test,]
  recommender = Recommender(data.train, method="UBCF",  parameter=list(method="Jaccard", nn=30))
  prediction = predict(recommender, newdata=data.test, n=5, type="topNList")
  result = c(result, getList(prediction,decode=TRUE,ratings=F))
}

result. = as.data.frame(result)
colnames(result.) = user
user = names(result) 
results= t(result.)

write.csv(results, "result.csv")

###분석완료


#검증

library(ggplot2)
library(recommenderlab)
vali = evaluationScheme(data=bin.rmatrix.exc, method="cross-validation", train=0.8, given=50, k=5, goodRating=1)
recommender = Recommender(data=getData(vali, "train"), method="UBCF",  parameter=list(method="Jaccard", nn=30))
prediction = predict(recommender, newdata=getData(vali, "known"), n=5, type="topNList", given=50)
accuracy11 = calcPredictionAccuracy(x=prediction11, data=getData(data.set1, "unknown"), byUser=F, given=50)


vali_accuracy = calcPredictionAccuracy(x=prediction, data=getData(vali, "unknown"), byUser=F, given=50)


eval = evaluate(x=vali, method="UBCF", n=c(3,5,seq(10,100,10)))

#평가

head(results)
head(result)
num = sapply(result, function(x) {goods$소분류명[which(x==goods$소분류코드)]})
head(num)

head(goods$소분류코드)
which(goods$소분류코드=="A040233")
which("A040233"==goods$소분류코드)
