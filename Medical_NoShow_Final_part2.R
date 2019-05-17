########Var, Max, Min of CV of different models#########

# first layer
var_model<-c(#var(crs$model_first_knn[["resample"]][["Accuracy"]]),
             var(crs$model_first_lr[["resample"]][["Accuracy"]]),
             var(crs$model_first_rpart2[["resample"]][["Accuracy"]]),
             var(crs$model_first_nnet[["resample"]][["Accuracy"]]))

mean_model<-c(#mean(crs$model_first_knn[["resample"]][["Accuracy"]]),
              mean(crs$model_first_lr[["resample"]][["Accuracy"]]),
              mean(crs$model_first_rpart2[["resample"]][["Accuracy"]]),
              mean(crs$model_first_nnet[["resample"]][["Accuracy"]]))

max_model<-c(#max(crs$model_first_knn[["resample"]][["Accuracy"]]),
             max(crs$model_first_lr[["resample"]][["Accuracy"]]),
             max(crs$model_first_rpart2[["resample"]][["Accuracy"]]),
             max(crs$model_first_nnet[["resample"]][["Accuracy"]]))

min_model<-c(#min(crs$model_first_knn[["resample"]][["Accuracy"]]),
             min(crs$model_first_lr[["resample"]][["Accuracy"]]),
             min(crs$model_first_rpart2[["resample"]][["Accuracy"]]),
             min(crs$model_first_nnet[["resample"]][["Accuracy"]]))
var_model
mean_model
max_model
min_model

# second layer
var(crs$model_second_rpart2[["resample"]][["Accuracy"]])
mean(crs$model_second_rpart2[["resample"]][["Accuracy"]])
max(crs$model_second_rpart2[["resample"]][["Accuracy"]])
min(crs$model_second_rpart2[["resample"]][["Accuracy"]])

########Predictors importance#########
# gbmImp_knn <- varImp(crs$model_first_knn, scale = FALSE)
gbmImp_lr <- varImp(crs$model_first_lr, scale = FALSE)
gbmImp_nnet <- varImp(crs$model_first_nnet, scale = FALSE)
gbmImp_rpart2 <- varImp(crs$model_first_rpart2, scale = FALSE)
# plot(gbmImp_knn, top = 5, main="knn")
plot(gbmImp_lr, top = 5, main="lr")
plot(gbmImp_nnet, top = 5, main="nnet")
plot(gbmImp_rpart2, top = 5, main="rpart2")

########Plot ROC#########

library(pROC)
# rocknn <- plot.roc(crs$dataset.cv_test$No_show, 
#                    predict(crs$model_first_knn, crs$dataset.cv_test, type = 'prob')$X1)
roclr <- plot.roc(crs$dataset.cv_test$No_show, 
                  predict(crs$model_first_lr, crs$dataset.cv_test, type = 'prob')$X1)
rocnnet <- plot.roc(crs$dataset.cv_test$No_show, 
                    predict(crs$model_first_nnet, crs$dataset.cv_test, type = 'prob')$X1)
rocrpart2 <- plot.roc(crs$dataset.cv_test$No_show, 
                      predict(crs$model_first_rpart2, crs$dataset.cv_test, type = 'prob')$X1)
rocrpart2Sec <- plot.roc(crs$dataset.cv_test$No_show, 
                         predict(crs$model_second_rpart2, crs$dataset.cv_test, type = 'prob')$X1)

# plot(rocknn,
#      type = "l", col="yellow",
#      legacy.axes = TRUE,
#      main="ROC"
# )
# par(new=TRUE) #continue plot

plot(roclr,
     type = "l", col ="red",
     legacy.axes = TRUE,
     axex = FALSE,
     xlab = "",
     ylab = "")
par(new=TRUE) #continue plot

plot(rocnnet,
     print.thres = c(.5),
     print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
     print.thres.cex = .8, 
     print.thres.col = "blue",
     type = "l", col ="blue",
     legacy.axes = TRUE,
     axex = FALSE,
     xlab = "",
     ylab = "")
par(new=TRUE) #continue plot

plot(rocrpart2,
     type = "l", col ="purple",
     legacy.axes = TRUE,
     axex = FALSE,
     xlab = "",
     ylab = "")
par(new=TRUE) #continue plot

plot(rocrpart2Sec,
     print.thres = c(.5),
     print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
     print.thres.cex = .8, 
     print.thres.col = "green",
     type = "l", col ="green",
     legacy.axes = TRUE,
     axex = FALSE,
     xlab = "",
     ylab = "")
par(new=TRUE) #continue plot

legend("bottomright", 
       legend = c("knn", "lr", "nnet","rpart2","rpart2Ens"), 
       col = c("yellow","red", "blue","purple","green"), 
       bty = "n", 
       lty = 1,
       pt.cex = 2, 
       cex = 1.2, 
       text.col = "black", 
       horiz = F , 
       inset = c(0.1, 0.1, 0.1, 0.1, 0.1,0.1))

