
#' Parzen window smoothing
#'
#' @param score anomaly score for which roc is computed
#' @param inx_anomaly indecies of abnormal variable
#'
#' @export

ROC <- function(score, inx_anomaly) {

    N <- length(score)

    xaxis     <- (0:N)/N
    ROC_value <- rep(0, N+1)

    binary <- rep(0, N)
    binary[inx_anomaly] <- 1

    # calculate ROC value
    score_order <- order(score, decreasing = TRUE)
    for (i in 1:N) {
        ROC_value[i+1] <- sum(binary[score_order[1:i]])/length(inx_anomaly)
    }

    # calculate AUC of ROC
    AUC <- 0
    for (i in 2:(N+1)) {
        AUC <- AUC + (ROC_value[i-1] + ROC_value[i]) * 1/N * 0.5
    }

    return (list(xaxis = xaxis, ROC_value = ROC_value, AUC = AUC))
}
